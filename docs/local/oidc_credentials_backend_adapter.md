# OIDC credential storage: pluggable backend adapter

Design note for **validmind-library** OIDC device-flow authentication. Describes the pluggable storage adapter so self-hosted and enterprise deployments can integrate approved secret stores (keychain, Vault, etc.) when local file caching does not meet their security requirements.

**Related code (validmind-library):**

- `validmind/credentials_store.py` — default `~/.validmind/credentials.json` persistence
- `validmind/oidc_device.py` — RFC 8628 device flow and token refresh
- `validmind/api_client.py` — `vm.init()` OIDC branch and `_obtain_oidc_tokens()`

**Related code (backend):**

- `src/backend/auth/auth.py` — `requires_client_auth` (library + public API Bearer vs API key)
- `src/backend/auth/oidc.py` — server-side userinfo cache in Redis (not client token persistence)

---

## Motivation

OIDC device flow (`vm.init(issuer=..., client_id=...)`) caches tokens locally for a smooth notebook experience. The default location is:

```
~/.validmind/credentials.json
```

That default uses restrictive POSIX permissions (`0700` directory, `0600` file, atomic write) and works well for single-user development. Some enterprise and self-hosted deployments need stronger options:

- Tokens stored in an **OS keychain** or **corporate secret manager** instead of a local JSON file
- **Memory-only** sessions on shared Jupyter or VM hosts (re-auth each process)
- **Custom persistence** aligned with internal security policies

This adapter is an **optional enterprise security enhancement**. The default file cache remains unchanged for users who do not configure a backend.

This applies to the **library client** only; the ValidMind backend API server does not persist client access tokens to disk.

---

## Current architecture

### Backend (server)

- Validates `Authorization: Bearer` JWTs on each request via `verify_auth_token`.
- Does **not** persist client access tokens to disk.
- Caches OIDC **userinfo** in Redis with a SHA-256 hash of the token as the cache key (TTL default 900s). Falls back to in-process LRU if Redis is unavailable.
- Library and public API connectivity policy: `api_key` | `device_flow` | `any` (vmconfig + per-org overrides).

### validmind-library (client)

1. `vm.init(..., issuer=, client_id=, ...)` calls `_obtain_oidc_tokens()`.
2. **Load** cached entry for `(issuer, client_id[, audience])`.
3. Reuse if not expired; **refresh** if refresh token present; otherwise run **device flow**.
4. **Persist** tokens via `credentials_store` (filesystem by default).
5. Send `Authorization: Bearer` on tracking API requests.

Public API sample script (`backend/scripts/sample_public_api_with_device_flow.py`) does not persist tokens — in-memory only for that run.

---

## Is filesystem caching standard?

**Common for developer CLIs/SDKs** (AWS CLI, Azure CLI, gcloud) and suitable as a default for local development.

Deployments with stricter policies often prefer the **OS credential store** (macOS Keychain, Windows Credential Manager, Linux Secret Service) or a **corporate secret manager**. The pluggable backend lets those deployments opt in without changing the default for everyone else.

---

## Proposed solution: `OidcCredentialsBackend` adapter

Provide a small **storage-only** interface that customers implement. The library retains ownership of OIDC discovery, device flow, refresh logic, and Bearer header selection.

### Principle

| Layer | Owner | Pluggable? |
|-------|-------|------------|
| Device flow, refresh, JWT usage | validmind-library | No |
| Token persistence (get / put / delete) | Customer or default file backend | **Yes** |

### Protocol (conceptual)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class OidcCredentialsBackend(Protocol):
    """Persist OIDC token entries keyed by credential_key(issuer, client_id, audience)."""

    def get(self, key: str) -> dict | None: ...
    def put(self, key: str, entry: dict) -> None: ...
    def delete(self, key: str) -> None: ...
```

**Entry dict contract** (unchanged from today):

| Field | Required | Notes |
|-------|----------|-------|
| `issuer`, `client_id` | yes | normalized |
| `access_token` | yes | |
| `expires_at` | yes | ISO-8601 UTC |
| `refresh_token` | no | enables silent refresh |
| `id_token` | no | Entra path may use this as Bearer |
| `audience` | no | when configured |

Public helpers remain shared utilities: `normalize_issuer`, `normalize_client_id`, `normalize_audience`, `credential_key`, `is_expired`.

### Built-in backends (ship in core)

| Backend | Purpose |
|---------|---------|
| `FileCredentialsBackend` | Current `~/.validmind/credentials.json` behavior — **default** |
| `MemoryCredentialsBackend` | No disk persistence; re-auth each process — shared Jupyter / CI |
| `NullCredentialsBackend` | Explicit alias for “never persist” |

Optional first-party extras (`validmind[keyring]`, etc.) can follow later; not required for the adapter story.

### Registration options

**1. Explicit (notebook / script)**

```python
import validmind as vm
from acme.validmind_credentials import VaultOidcBackend

vm.init(
    issuer="...",
    client_id="...",
    model="...",
    credentials_backend=VaultOidcBackend(vault_path="..."),
)
```

**2. Environment / platform injection (JupyterHub, VM image)**

```bash
export VM_OIDC_CREDENTIALS_BACKEND="acme.validmind_credentials:VaultOidcBackend"
export VM_OIDC_CREDENTIALS_BACKEND_KWARGS='{"vault_path":"..."}'
```

**3. IPython / site startup hook**

```python
import validmind.credentials_backend as cb
cb.set_default_backend(VaultOidcBackend(...))
```

Env-based registration is important for self-hosted Jupyter where end users do not write adapter code.

---

## Design decisions to nail down

- **Thread safety** — Backends may be called from sync paths and future async refresh; document process-safe requirements.
- **Multi-tenant Jupyter** — Allow backends to scope storage by `$JUPYTERHUB_USER`, org id, etc. (constructor kwargs from env or context dict).
- **Errors** — `get` returns `None` when missing; storage failures raise `ValidMindAuthError` (do not silently fall back to device flow on Vault outage unless configured).
- **Logout / clear** — Expose `vm.clear_oidc_credentials()` delegating to `backend.delete(key)`.
- **Migration** — Optional one-time read from file backend and re-put to custom backend when switching deployments.

---

## Suggested implementation rollout (validmind-library)

1. Extract `FileCredentialsBackend` from `credentials_store.py` (no behavior change).
2. Add `MemoryCredentialsBackend` and env-based backend loading.
3. Thread `credentials_backend` through `init()` and `_obtain_oidc_tokens()`.
4. Document the protocol and a minimal reference adapter (e.g. env-var-only tokens for CI).
5. Update `docs/oidc-device-flow-release-notes.md` with enterprise / adapter section.
6. Optionally publish reference adapters (Keychain, Vault) in a separate template or extras package.

---

## Customer-facing summary (security questionnaire)

> **Default:** OIDC access and refresh tokens are stored locally on the analyst machine at `~/.validmind/credentials.json` with user-only file permissions. Tokens are not stored on ValidMind servers. This default is intended for individual development and notebook use.
>
> **Enterprise (optional):** Organizations with stricter credential policies can use a pluggable **`OidcCredentialsBackend`** to persist tokens in an approved system (OS keychain, HashiCorp Vault, AWS Secrets Manager, etc.). The SDK continues to handle OIDC device flow and token refresh; only persistence is delegated to your adapter. Set `VM_OIDC_NO_PERSIST=1` for memory-only storage on shared hosts.
>
> **Alternative:** Use API key authentication (`VM_API_KEY` / `VM_API_SECRET`) with secrets managed by your existing tooling — already supported and often preferred in high-assurance environments.

---

## Out of scope (for this adapter)

- Pluggable OAuth grant types (authorization code, client credentials) — device flow only today.
- Backend server changes — server already does not write client tokens to disk.
- Shipping every secret-manager integration inside core `validmind` — customer-owned adapters instead.
