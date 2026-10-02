# ValidMind Tracking Core

`validmind-tracking-core` contains the dependency-light authentication and tracking
transport shared by ValidMind SDKs. It supports API-key authentication and OIDC
device-flow authentication without importing the full `validmind` package.

This package is an implementation dependency of ValidMind SDKs. Application code
should normally use `validmind-metrics` or `validmind` directly.
