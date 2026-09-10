# ValidMind Metrics

`validmind-metrics` is a lightweight client for sending unit metrics to the
ValidMind Platform. It supports API-key authentication and OIDC device-flow
authentication without installing or importing the full `validmind` library.

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

OIDC authentication uses the same `issuer`, `client_id`, optional `scope`, and
optional `audience` settings as the full library. Cached credentials are stored
under `~/.validmind/credentials.json`.

For an async HTTP handler, use `await client.alog_metric(...)` so the blocking
transport runs outside the event loop.
