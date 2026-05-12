# ValidMind Python library: OIDC device flow (release notes)

This note summarizes what your identity provider (IdP) must expose, how to call the library with the device authorization flow ([RFC 8628](https://datatracker.ietf.org/doc/html/rfc8628)), and how authentication works end to end. Share it with security and platform teams when enabling notebook or CLI login without long-lived API keys.

---

## 1. IdP configuration the library depends on

The SDK does **OIDC discovery** against your issuer. From `{issuer}/.well-known/openid-configuration` it **requires** at least:

| Discovery field | Why |
|-----------------|-----|
| `device_authorization_endpoint` | Start the device flow (user gets a code and verification URL). |
| `token_endpoint` | Poll for tokens after authorization, and refresh access tokens. |

**OAuth / IdP app registration**

- Register an OAuth **public client** used only with the device flow (no embedded client secret in the library).
- Enable the **device code** / device authorization grant for that client (names vary by vendor).  
  - **Microsoft Entra ID:** Turn on **Allow public client flows** for the app registration and ensure the device code flow is permitted for your tenant/policy.
- Supply ValidMind with values that match what the **ValidMind platform** already trusts for your organization:
  - **Issuer URL** — Normalized OpenID Provider issuer (example Entra v2: `https://login.microsoftonline.com/<tenant-id>/v2.0`).
  - **Client ID** — The application (client) ID of the public OAuth client.
  - **Scopes** — The library defaults to `openid profile email`. Your IdP may require **additional scopes** or a **resource identifier** so the access token is acceptable to the ValidMind API.
  - **Audience / API identifier (often required)** — Access tokens must pass ValidMind’s JWT validation (issuer, audience, signing keys). Many setups need an explicit **audience** for API-style access tokens (e.g. Auth0 API Identifier, or Azure AD custom scope / resource). Pass this as the `audience` argument (or set env `VM_OIDC_AUDIENCE`) so the provider issues tokens ValidMind can verify.

**Operational checks**

- Misconfiguration often appears as a token issued by the IdP but **401/403 from ValidMind** (e.g. wrong audience). Align `audience` / scopes with what ValidMind has configured for your tenant (`api_audience` and JWKS).
- Revoking access: users can sign out or revoke the app in the IdP portal; the library also caches tokens locally (see below).

---

## 2. `vm.init` and `api_client.init` (same API)

`import validmind as vm` exposes **`vm.init`**, which is the same function as **`validmind.api_client.init`**. There is no separate “OIDC-only” entrypoint—authentication mode is selected by the arguments (and environment variables) you pass.

**API key mode (unchanged)**

```python
vm.init(
    api_key="...",
    api_secret="...",
    api_host="https://.../api/v1/tracking/",  # or api_url= (alias)
    model="<model-cuid>",
)
```

Environment variables still apply when arguments are omitted: `VM_API_KEY`, `VM_API_SECRET`, `VM_API_HOST`, `VM_API_MODEL`.

**OIDC device flow mode (new)**

```python
vm.init(
    issuer="https://login.microsoftonline.com/<tenant>/v2.0",
    client_id="<oauth-public-client-id>",
    model="<model-cuid>",
    api_host="https://.../api/v1/tracking/",  # or api_url= (alias); defaults from VM_API_HOST if unset
    scope="openid profile email",  # optional; this is the default if omitted
    audience="<resource-or-api-identifier>",  # optional; often required for API tokens; or VM_OIDC_AUDIENCE
)
```

Rules:

- Provide **either** API key + secret **or** OIDC (`issuer` + `client_id`). Mixing both raises an error.
- If `issuer` is set, **`client_id` is required** (and vice versa).
- Optional **`audience`** can also be set via **`VM_OIDC_AUDIENCE`** if you prefer not to put it in code.

Equivalent import:

```python
from validmind import api_client

api_client.init(issuer="...", client_id="...", model="...", api_host="...")
```

---

## 3. How the flow works (overview)

1. **`vm.init(..., issuer=, client_id=, ...)`** loads cached tokens from `~/.validmind/credentials.json` (if present) for the normalized `(issuer, client_id)` — and optional audience — key.
2. If there is a **valid access token**, it is reused.
3. If the access token is expired but a **refresh token** is available, the library **refreshes** silently; on failure it clears that cache entry and continues.
4. If no usable token exists, the library runs the **device authorization flow**:
   - Fetches OIDC metadata from `{issuer}/.well-known/openid-configuration`.
   - POSTs to **`device_authorization_endpoint`** with `client_id`, `scope`, and `audience` (when configured).
   - Prints **verification URL** and **user code** (from the IdP response) so the user can complete login in a browser.
   - **Polls** **`token_endpoint`** until the user authorizes, using `grant_type` for the device code per RFC 8628, then stores **access** (and refresh when provided) tokens in the credential file.
5. Subsequent calls to the ValidMind **tracking API** send **`Authorization: Bearer <access_token>`** together with the usual headers such as **`X-MODEL-CUID`**. API key headers are not used in OIDC mode.

Org and model access are enforced server-side: the user must be allowed to use the model identified by `model` (model CUID), consistent with ValidMind’s existing authorization rules for library clients.

---

## Quick reference

| Item | Purpose |
|------|---------|
| Issuer URL | Discovery and token validation context |
| Client ID | Public OAuth client for device flow |
| Model CUID | Identifies the inventory model (`model=` or `VM_API_MODEL`) |
| API host / URL | Tracking API base (`api_host`, `api_url`, or `VM_API_HOST`) |
| Scope | OAuth scopes (default `openid profile email`) |
| Audience | Often needed so access tokens target the ValidMind API (`audience` or `VM_OIDC_AUDIENCE`) |
| Credential file | `~/.validmind/credentials.json` (cached tokens; restrict like other secrets on shared machines) |

For implementation details in this repository, see `validmind/oidc_device.py`, `validmind/credentials_store.py`, and `validmind/api_client.py`.
