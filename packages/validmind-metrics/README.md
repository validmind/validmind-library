# ValidMind Metrics

`validmind-metrics` is a lightweight client for sending unit metrics to the
ValidMind Platform. It supports API-key authentication and OIDC device-flow
authentication without installing or importing the full `validmind` library.

## Module-level API

```python
import validmind_metrics

validmind_metrics.init()  # optional; reads the VM_* environment variables below
validmind_metrics.log_metric("accuracy", 0.95)
await validmind_metrics.alog_metric("accuracy", 0.95)  # from async code
```

If `init()` is not called, the first `log_metric` / `alog_metric` call creates
the default client from the environment. `init(**kwargs)` accepts the same
arguments as `MetricsClient`.

## Explicit client

```python
from validmind_metrics import MetricsClient

client = MetricsClient(
    api_host="https://app.validmind.ai/api/v1/tracking",
    model="model-cuid",
    api_key="api-key",
    api_secret="api-secret",
)

client.log_metric("accuracy", 0.95)
```

## Environment variables

| Variable | Meaning |
| --- | --- |
| `VM_API_HOST` / `VM_API_URL` | Tracking API URL |
| `VM_API_MODEL` | Model CUID |
| `VM_API_KEY`, `VM_API_SECRET` | API-key credentials |
| `VM_OIDC_ISSUER`, `VM_OIDC_CLIENT_ID` | OIDC credentials (instead of an API key) |
| `VM_OIDC_SCOPE`, `VM_OIDC_AUDIENCE` | Optional OIDC scope and audience |
| `VM_API_TIMEOUT` | Request timeout in seconds (default 30) |

Explicit arguments take precedence over environment variables.

## Authentication in services

Use API-key credentials for long-running services and HTTP handlers.

OIDC uses the same `issuer`, `client_id`, optional `scope`, and optional
`audience` settings as the full library, and caches tokens in
`~/.validmind/credentials.json`. The client is non-interactive by default: with
no usable cached token it raises `TrackingAuthError` rather than waiting for a
device login. To log in, run once from a terminal with
`init(interactive=True)` (or `MetricsClient(..., interactive=True)`); later
processes reuse and refresh the cached token.

## Async code

`await alog_metric(...)` runs the blocking HTTP request in the event loop's
default thread-pool executor, so the loop itself is not blocked. Each call
holds one executor thread for a full round trip (up to `VM_API_TIMEOUT`); there
is no background queue. Calling the synchronous `log_metric` from a coroutine
blocks the loop for the duration of the request.

## Errors

All SDK errors derive from `TrackingError`: `TrackingConfigurationError`,
`TrackingAuthError`, `TrackingConnectionError` (network failure or timeout),
and `TrackingAPIError` (the API rejected the request). Invalid metric
arguments raise `ValueError`.
