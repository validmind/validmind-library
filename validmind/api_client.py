# Copyright © 2023-2026 ValidMind Inc. All rights reserved.
# Refer to the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

"""ValidMind API client

Note that this takes advantage of the fact that python modules are singletons to store and share
the configuration and session across the entire project regardless of where the client is imported.
"""
import asyncio
import atexit
import json
import os
import threading
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode, urljoin

import aiohttp
import requests
from aiohttp import FormData
from validmind_tracking_core.errors import TrackingAPIError
from validmind_tracking_core.metrics import post_metric, serialize_metric

from .__version__ import __version__
from .client_config import client_config
from .errors import (
    MissingAPICredentialsError,
    MissingModelIdError,
    ValidMindAuthError,
    raise_api_error,
)
from .logging import get_logger, log_api_operation
from .utils import NumpyEncoder, is_html, md_to_html, run_async
from .vm_models.figure import Figure

logger = get_logger(__name__)

_api_key = os.getenv("VM_API_KEY")
_api_secret = os.getenv("VM_API_SECRET")
_api_host = os.getenv("VM_API_HOST")
_model_cuid = os.getenv("VM_API_MODEL")
_document = None
_monitoring = False
_auth_mode = "api_key"
_access_token: Optional[str] = None
_oidc_login_context: Optional[Dict[str, str]] = None
# Expiry (ISO-8601) of the in-memory OIDC access token, for cheap request-path
# staleness checks; guarded together with the token by _oidc_refresh_lock.
_oidc_expires_at: Optional[str] = None
_oidc_refresh_lock = threading.Lock()

__api_session: Optional[aiohttp.ClientSession] = None


def _invalidate_async_session() -> None:
    """Drop the aiohttp session so the next request picks up new headers."""
    global __api_session
    sess = __api_session
    __api_session = None
    if sess is None or sess.closed:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(sess.close())
    except RuntimeError:
        try:
            asyncio.run(sess.close())
        except RuntimeError:
            pass


@atexit.register
def _close_session():
    """Closes the async client session at exit."""
    if __api_session and not __api_session.closed:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(__api_session.close())
            else:
                loop.run_until_complete(__api_session.close())
        except RuntimeError as e:
            # ignore RuntimeError when closing the session from the main thread
            if "no current event loop in thread" in str(e):
                pass
            elif "Event loop is closed" in str(e):
                pass
            else:
                raise e
        except Exception as e:
            logger.exception("Error closing aiohttp session at exit: %s", e)


def get_api_host() -> Optional[str]:
    return _api_host


def get_api_model() -> Optional[str]:
    return _model_cuid


def _get_api_headers() -> Dict[str, str]:
    headers = {
        "X-MODEL-CUID": _model_cuid,
        "X-MONITORING": str(_monitoring),
        "X-LIBRARY-VERSION": __version__,
    }
    if _document:
        headers["X-DOCUMENT-TYPE"] = _document
    if _auth_mode == "oidc":
        if not _access_token:
            raise ValidMindAuthError(
                "OAuth access token is missing. Run vm.init() again with issuer and client_id."
            )
        headers["Authorization"] = f"Bearer {_access_token}"
    else:
        headers["X-API-KEY"] = _api_key
        headers["X-API-SECRET"] = _api_secret
    return headers


def _get_session() -> aiohttp.ClientSession:
    """Initializes the async client session."""
    global __api_session

    if not __api_session or __api_session.closed:
        __api_session = aiohttp.ClientSession(
            headers=_get_api_headers(),
            timeout=aiohttp.ClientTimeout(total=int(os.getenv("VM_API_TIMEOUT", 30))),
            trust_env=True,
        )

    return __api_session


def _get_url(
    endpoint: str,
    params: Optional[Dict[str, str]] = None,
) -> str:
    global _api_host

    params = params or {}

    if not _api_host.endswith("/"):
        _api_host += "/"

    if params:
        return f"{urljoin(_api_host, endpoint)}?{urlencode(params)}"

    return urljoin(_api_host, endpoint)


async def _get(
    endpoint: str, params: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    _ensure_fresh_oidc_token()
    url = _get_url(endpoint, params)
    session = _get_session()

    async with session.get(url) as r:
        # A 401 can still slip through if the token was revoked ahead of its
        # stated expiry; force a refresh and retry once (safe — GET has no body).
        if r.status == 401 and _ensure_fresh_oidc_token(force=True):
            session = _get_session()
            async with session.get(url) as retry:
                if retry.status != 200:
                    _raise_for_api_error(retry.status, await retry.text())
                return await retry.json()
        if r.status != 200:
            _raise_for_api_error(r.status, await r.text())

        return await r.json()


async def _post(
    endpoint: str,
    params: Optional[Dict[str, str]] = None,
    data: Optional[Union[dict, FormData]] = None,
    files: Optional[Dict[str, Tuple[str, BytesIO, str]]] = None,
) -> Dict[str, Any]:
    _ensure_fresh_oidc_token()
    url = _get_url(endpoint, params)
    session = _get_session()

    if not isinstance(data, (dict)) and files is not None:
        raise ValueError("Cannot pass both non-json data and file objects to _post")

    if files:
        _data = FormData()

        for key, value in (data or {}).items():
            _data.add_field(key, value)

        for key, file_info in (files or {}).items():
            _data.add_field(
                key,
                file_info[1],
                filename=file_info[0],
                content_type=file_info[2] if len(file_info) > 2 else None,
            )
    else:
        _data = data

    async with session.post(url, data=_data) as r:
        # Not retried on 401: the request body / upload stream may already be
        # consumed. Proactive refresh above covers the expiry case; a 401 here
        # means a genuinely rejected token, surfaced as a clear auth error.
        if r.status != 200:
            _raise_for_api_error(r.status, await r.text())

        return await r.json()


def _ping() -> Dict[str, Any]:
    """Validates that we can connect to the ValidMind API (does not use the async session)."""
    _ensure_fresh_oidc_token()
    r = requests.get(
        url=_get_url("ping"),
        headers=_get_api_headers(),
    )
    if r.status_code == 401 and _ensure_fresh_oidc_token(force=True):
        r = requests.get(
            url=_get_url("ping"),
            headers=_get_api_headers(),
        )
    if r.status_code != 200:
        _raise_for_api_error(r.status_code, r.text)

    client_info = r.json()

    # Sentry removed: no telemetry initialization

    # Only show this confirmation the first time we connect to the API
    ack_connected = not client_config.model

    client_config.documentation_template = client_info.get("documentation_template", {})
    client_config.feature_flags = client_info.get("feature_flags", {})
    client_config.model = client_info.get("model", {})
    client_config.document_type = client_info.get(
        "document_type", "model_documentation"
    )

    if ack_connected:
        logger.info(
            f"🎉 Connected to ValidMind!\n"
            f"📊 Model: {client_config.model.get('name', 'N/A')} "
            f"(ID: {client_config.model.get('cuid', 'N/A')})\n"
            f"📁 Document Type: {client_config.document_type}"
        )


def _refresh_oidc_from_cache(
    issuer: str,
    client_id: str,
    refresh_token: str,
    scope: Optional[str],
    audience: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Exchange a cached refresh token for new tokens, persisting the result.

    Shared by the ``init()`` path (``_obtain_oidc_tokens``) and the request path
    (``_ensure_fresh_oidc_token``) so their refresh behaviour can't drift. On
    success the refreshed entry is cached and returned. On a failed refresh (e.g.
    revoked token / ``invalid_grant``) the stale entry is deleted and None is
    returned, so a dead refresh token isn't re-attempted; the caller decides how
    to recover (``init()`` falls back to device flow, the request path stops).
    """
    from .credentials_store import delete_cached_entry, upsert_cached_entry
    from .oidc_device import try_refresh_cached_tokens

    try:
        new_tokens = try_refresh_cached_tokens(
            issuer, client_id, refresh_token, scope, audience=audience
        )
    except ValidMindAuthError:
        delete_cached_entry(issuer, client_id, audience=audience)
        return None
    upsert_cached_entry(issuer, client_id, new_tokens, audience=audience)
    return new_tokens


def _obtain_oidc_tokens(
    issuer: str,
    client_id: str,
    scope: str,
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a credentials entry dict with access_token, expires_at, refresh_token, etc."""
    from .credentials_store import (
        get_cached_entry,
        is_expired,
        normalize_client_id,
        normalize_issuer,
        upsert_cached_entry,
    )
    from .oidc_device import run_device_flow

    norm_issuer = normalize_issuer(issuer)
    norm_client_id = normalize_client_id(client_id)
    cached = get_cached_entry(norm_issuer, norm_client_id, audience=audience)
    if cached and not is_expired(cached):
        return cached
    if cached and cached.get("refresh_token"):
        new_tokens = _refresh_oidc_from_cache(
            norm_issuer, norm_client_id, cached["refresh_token"], scope, audience
        )
        if new_tokens is not None:
            return new_tokens
    tokens = run_device_flow(norm_issuer, norm_client_id, scope, audience=audience)
    upsert_cached_entry(norm_issuer, norm_client_id, tokens, audience=audience)
    return tokens


def _is_entra_issuer(issuer: str) -> bool:
    return "login.microsoftonline.com" in issuer.lower()


def _select_oidc_bearer_token(entry: Dict[str, Any]) -> str:
    if _is_entra_issuer(entry.get("issuer", "")) and entry.get("id_token"):
        return entry["id_token"]
    return entry["access_token"]


def _set_oidc_access_token(entry: Dict[str, Any]) -> None:
    """Adopt the bearer token from a credentials entry as the active token.

    Records its expiry for cheap request-path staleness checks and drops the
    pooled aiohttp session so the next request rebuilds it with the new token.
    """
    global _access_token, _oidc_expires_at
    _access_token = _select_oidc_bearer_token(entry)
    _oidc_expires_at = entry.get("expires_at")
    _invalidate_async_session()


def _clear_oidc_access_token() -> None:
    """Drop the in-memory OIDC token and pooled session after an unrecoverable refresh.

    Pairs with deleting the cached entry: with no usable token in memory, the next
    request fails fast at header build with the clear "run vm.init()" error instead
    of sending a doomed request just to get a 401 back.
    """
    global _access_token, _oidc_expires_at
    _access_token = None
    _oidc_expires_at = None
    _invalidate_async_session()


def _oidc_token_is_stale() -> bool:
    """Cheap in-memory expiry check (no file I/O) using the 120s skew in is_expired."""
    from .credentials_store import is_expired

    return is_expired({"expires_at": _oidc_expires_at})


def _ensure_fresh_oidc_token(force: bool = False) -> bool:
    """Refresh the OIDC access token on the request path when it has expired.

    The token captured at ``init()`` is otherwise reused unchanged for the life of
    the process, so a session that outlives the provider's access-token lifetime
    starts failing on every call until ``init()`` is re-run. This uses the cached
    refresh token (requires ``offline_access``, the default scope) to obtain a new
    access token in place. Returns True when a usable token is in place, False when
    no refresh was possible (non-OIDC mode, no cached refresh token, or a failed
    refresh — in which case the cached entry is dropped, matching init()).
    """
    if _auth_mode != "oidc" or _oidc_login_context is None:
        return False
    # Hot path: skip file I/O and the lock while the current token is still valid.
    if not force and not _oidc_token_is_stale():
        return True

    from .credentials_store import get_cached_entry, is_expired

    ctx = _oidc_login_context
    issuer = ctx["issuer"]
    client_id = ctx["client_id"]
    scope = ctx.get("scope")
    audience = ctx.get("audience") or None

    with _oidc_refresh_lock:
        entry = get_cached_entry(issuer, client_id, audience=audience)
        if entry is None:
            return False
        # Another caller may have refreshed the token while we waited on the lock.
        if not force and not is_expired(entry):
            _set_oidc_access_token(entry)
            return True
        refresh_token = entry.get("refresh_token")
        if not refresh_token:
            return False
        # Shared with init(): on failure the cached entry is dropped so a
        # bad/revoked refresh token isn't re-attempted on every request. Recovery
        # is re-running vm.init() (surfaced by the clear 401 auth error); unlike
        # init() the request path can't fall back to interactive device flow.
        new_tokens = _refresh_oidc_from_cache(
            issuer, client_id, refresh_token, scope, audience
        )
        if new_tokens is None:
            # Hard failure: the cache entry was dropped; clear the in-memory token
            # too so the next request fails fast with the clear re-auth message.
            _clear_oidc_access_token()
            return False
        _set_oidc_access_token(new_tokens)
        return True


def _raise_for_api_error(status: int, text: str) -> None:
    """Raise a clear, actionable auth error for an OIDC 401, else defer to raise_api_error."""
    if status == 401 and _auth_mode == "oidc":
        raise ValidMindAuthError(
            "ValidMind rejected the OAuth access token (HTTP 401); it may have "
            "expired or been revoked. Run vm.init() again to re-authenticate."
        )
    raise_api_error(text)


def init(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_host: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    monitoring: bool = False,
    generate_descriptions: Optional[bool] = None,
    document: Optional[str] = None,
    issuer: Optional[str] = None,
    client_id: Optional[str] = None,
    scope: Optional[str] = None,
    audience: Optional[str] = None,
):
    """
    Initializes the API client instances and calls the /ping endpoint to ensure
    the provided credentials are valid and we can connect to the ValidMind API.

    If the API key and secret are not provided, the client will attempt to
    retrieve them from the environment variables `VM_API_KEY` and `VM_API_SECRET`.

    Alternatively, pass ``issuer`` and ``client_id`` or set their ``VM_OIDC_*``
    environment variables to authenticate via the OIDC device authorization flow
    (RFC 8628). Tokens are cached under ``~/.validmind/credentials.json``. Do not
    combine API keys with OIDC parameters.

    Args:
        model (str, optional): The model CUID. Defaults to None.
        api_key (str, optional): The API key. Defaults to None.
        api_secret (str, optional): The API secret. Defaults to None.
        api_host (str, optional): The API host (tracking base URL). Defaults to
            env ``VM_API_HOST`` or ``VM_API_URL``.
        api_url (str, optional): Alias for ``api_host``. Defaults to env
            ``VM_API_URL`` or ``VM_API_HOST``.
        monitoring (bool): The ongoing monitoring flag. Defaults to False.
        generate_descriptions (bool, optional): Whether to use GenAI to generate test result descriptions. Defaults to True.
        document (str, optional): The name of the document. Omitting this argument is deprecated.
        issuer (str, optional): OIDC issuer URL (e.g. Entra tenant ``.../v2.0``).
            Can be set via env ``VM_OIDC_ISSUER``.
        client_id (str, optional): OAuth public client id for device flow. Can be
            set via env ``VM_OIDC_CLIENT_ID``.
        scope (str, optional): OAuth scopes (default ``openid profile email offline_access``).
            Can be set via env ``VM_OIDC_SCOPE``.
        audience (str, optional): Resource / API identifier for the access token
            (e.g. Auth0 API Identifier). Use the same value the ValidMind backend
            expects as ``api_audience`` so the provider can issue RS256 API tokens.
            Can be set via env ``VM_OIDC_AUDIENCE``.

    Raises:
        MissingAPICredentialsError: If neither API keys nor OIDC parameters can be resolved.
        MissingModelIdError: If model id is missing.
        ValidMindAuthError: If OIDC configuration conflicts or login fails.
    """
    global _api_key, _api_secret, _api_host, _model_cuid, _monitoring, _document
    global _auth_mode, _access_token, _oidc_login_context, _oidc_expires_at

    if api_key == "...":
        # special case to detect when running a notebook placeholder (...)
        # will override with environment variables for easier local development
        api_host = api_url = api_key = api_secret = model = issuer = client_id = (
            audience
        ) = None

    _model_cuid = model or os.getenv("VM_API_MODEL")
    if _model_cuid is None:
        raise MissingModelIdError()

    resolved_host = (
        api_url
        or api_host
        or os.getenv("VM_API_URL")
        or os.getenv("VM_API_HOST", "http://localhost:5000/api/v1/tracking/")
    )
    oidc_issuer = issuer if issuer is not None else os.getenv("VM_OIDC_ISSUER")
    oidc_client_id = (
        client_id if client_id is not None else os.getenv("VM_OIDC_CLIENT_ID")
    )
    oidc_scope = scope if scope is not None else os.getenv("VM_OIDC_SCOPE")

    env_key = api_key if api_key is not None else os.getenv("VM_API_KEY")
    env_secret = api_secret if api_secret is not None else os.getenv("VM_API_SECRET")
    has_api_creds = (
        env_key is not None
        and env_secret is not None
        and env_key != ""
        and env_secret != ""
    )
    has_oidc = bool(oidc_issuer and oidc_client_id)
    if has_oidc and has_api_creds:
        raise ValidMindAuthError(
            "Provide either API credentials (api_key and api_secret) or OIDC "
            "(issuer and client_id), not both."
        )
    if oidc_issuer and not oidc_client_id:
        raise ValidMindAuthError("client_id is required when issuer is set.")
    if oidc_client_id and not oidc_issuer:
        raise ValidMindAuthError("issuer is required when client_id is set.")

    _monitoring = monitoring

    if generate_descriptions is not None:
        os.environ["VALIDMIND_LLM_DESCRIPTIONS_ENABLED"] = str(generate_descriptions)

    if document is None:
        logger.error(
            "Future releases will require `document` as one of the options you must provide to `vm.init()`. "
            "To learn more, refer to https://docs.validmind.ai/developer/validmind-library.html"
        )

    _document = document

    if has_oidc:
        _auth_mode = "oidc"
        _api_key = None
        _api_secret = None
        _api_host = resolved_host
        scope_val = oidc_scope or "openid profile email offline_access"
        from .credentials_store import normalize_audience

        oidc_audience_val = normalize_audience(
            audience if audience is not None else os.getenv("VM_OIDC_AUDIENCE")
        )
        oidc_audience_opt = oidc_audience_val or None
        entry = _obtain_oidc_tokens(
            oidc_issuer, oidc_client_id, scope_val, audience=oidc_audience_opt
        )
        _oidc_login_context = {
            "issuer": entry["issuer"],
            "client_id": entry["client_id"],
            "scope": scope_val,
            "audience": entry.get("audience") or oidc_audience_val,
        }
        # Sets _access_token / _oidc_expires_at and invalidates the async session.
        _set_oidc_access_token(entry)
    else:
        if env_key is None or env_secret is None:
            raise MissingAPICredentialsError()
        _auth_mode = "api_key"
        _access_token = None
        _oidc_login_context = None
        _oidc_expires_at = None
        _api_key = env_key
        _api_secret = env_secret
        _api_host = resolved_host
        _invalidate_async_session()

    reload()


def reload():
    """Reconnect to the ValidMind API and reload the project configuration."""
    _ping()


async def aget_metadata(content_id: str) -> Dict[str, Any]:
    """Gets a metadata object from ValidMind API.

    Args:
        content_id (str): Unique content identifier for the metadata.

    Raises:
        Exception: If the API call fails.

    Returns:
        dict: Metadata object.
    """
    return await _get(f"get_metadata/{content_id}")


async def alog_metadata(
    content_id: str,
    text: Optional[str] = None,
    _json: Optional[Dict[str, Any]] = None,
    section_id: Optional[str] = None,
    text_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Logs free-form metadata to ValidMind API.

    Args:
        content_id (str): Unique content identifier for the metadata.
        text (str, optional): Free-form text to assign to the metadata. Defaults to None.
        _json (dict, optional): Free-form key-value pairs to assign to the metadata. Defaults to None.
        section_id (str, optional): Section ID to append the text block to when the
            content ID does not already exist.
        text_format (str, optional): Format of ``text``. Markdown is sent as-is when
            the backend advertises support so conversion happens after the request
            passes through the WAF. Older backends receive locally converted HTML.

    Raises:
        Exception: If the API call fails.

    Returns:
        dict: The response from the API.
    """
    if text_format == "markdown" and not client_config.supports_log_metadata_markdown():
        text = md_to_html(text, mathml=True)
        text_format = None

    metadata_dict = {"content_id": content_id}
    if text is not None:
        metadata_dict["text"] = text
    if _json is not None:
        metadata_dict["json"] = _json
    if text_format is not None:
        metadata_dict["text_format"] = text_format

    request_params = {}
    if section_id:
        request_params["section_id"] = section_id

    try:
        return await _post(
            "log_metadata",
            params=request_params,
            data=json.dumps(metadata_dict, cls=NumpyEncoder, allow_nan=False),
        )
    except Exception as e:
        logger.error("Error logging metadata to ValidMind API")
        raise e


@log_api_operation(
    operation_name="Sending figure to ValidMind API",
    extract_key=lambda figure: figure.key,
)
async def alog_figure(figure: Figure) -> Dict[str, Any]:
    """Logs a figure.

    Args:
        figure (Figure): The Figure object wrapper.

    Raises:
        Exception: If the API call fails.

    Returns:
        dict: The response from the API.
    """
    try:
        return await _post(
            "log_figure",
            data=figure.serialize(),
            files=figure.serialize_files(),
        )
    except Exception as e:
        logger.error("Error logging figure to ValidMind API")
        raise e


async def alog_test_result(
    result: Dict[str, Any],
    section_id: str = None,
    position: int = None,
    unsafe: bool = False,
    config: Dict[str, bool] = None,
) -> Dict[str, Any]:
    """Logs test results information.

    This method will be called automatically from any function running tests but
    can also be called directly if the user wants to run tests on their own.

    Args:
        result (dict): A dictionary representing the test result.
        section_id (str, optional): The section ID add a test driven block to the documentation.
        position (int): The position in the section to add the test driven block.

    Raises:
        Exception: If the API call fails.

    Returns:
        dict: The response from the API.
    """
    request_params = {}
    if section_id:
        request_params["section_id"] = section_id
    if position is not None:
        request_params["position"] = position
    try:
        return await _post(
            "log_test_results",
            params=request_params,
            data=json.dumps(
                {**result, "config": config},
                cls=NumpyEncoder,
                allow_nan=False,
            ),
        )
    except Exception as e:
        logger.error("Error logging test results to ValidMind API")
        raise e


def log_test_result(
    result: Dict[str, Any],
    section_id: str = None,
    position: int = None,
    unsafe: bool = False,
    config: Dict[str, bool] = None,
) -> Dict[str, Any]:
    """Logs test results information.

    Args:
        result (dict): A dictionary representing the test result.
        section_id (str, optional): The section ID add a test driven block to the documentation.
        position (int): The position in the section to add the test driven block.
        unsafe (bool): If True, log the result even if it contains sensitive data.
        config (Dict[str, bool]): Configuration options for displaying the test result.

    Returns:
        dict: The response from the API.
    """
    return run_async(
        alog_test_result,
        result=result,
        section_id=section_id,
        position=position,
        unsafe=unsafe,
        config=config,
    )


async def alog_input(
    input_id: str, type: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Logs input information - internal use for now (don't expose via public API)

    Args:
        input_id (str): The input_id of the input
        type (str): The type of the input
        metadata (dict): The metadata of the input

    Raises:
        Exception: If the API call fails

    Returns:
        dict: The response from the API
    """
    try:
        return await _post(
            "log_input",
            data=json.dumps(
                {
                    "name": input_id,
                    "type": type,
                    "metadata": metadata,
                },
                cls=NumpyEncoder,
                allow_nan=False,
            ),
        )
    except Exception as e:
        logger.error("Error logging input to ValidMind API")
        raise e


def log_input(input_id: str, type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return run_async(alog_input, input_id, type, metadata)


def _validate_log_text_context(
    context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, List[str]]]:
    """Validate supported AI generation context for ``log_text``."""
    if context is None:
        return None

    if not isinstance(context, dict):
        raise ValueError("`context` must be a dictionary or None")

    allowed_keys = {"content_ids"}
    unknown_keys = set(context.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(
            "Unsupported `context` keys: "
            f"{', '.join(sorted(unknown_keys))}. Only `content_ids` is supported."
        )

    content_ids = context.get("content_ids")
    if content_ids is None:
        raise ValueError("`context` must include `content_ids` when provided")
    if not isinstance(content_ids, list) or not content_ids:
        raise ValueError("`context['content_ids']` must be a non-empty list")
    if any(
        not isinstance(content_id, str) or not content_id for content_id in content_ids
    ):
        raise ValueError("`context['content_ids']` must contain only non-empty strings")

    return {"content_ids": content_ids}


def generate_qualitative_text(text_generation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate qualitative text using the ValidMind AI API."""
    _ensure_fresh_oidc_token()
    r = requests.post(
        url=_get_url("ai/generate/qualitative_text_generation"),
        headers=_get_api_headers(),
        json=text_generation_data,
    )

    if r.status_code != 200:
        _raise_for_api_error(r.status_code, r.text)

    return r.json()


def _validate_logged_text(text: str, field_name: str) -> str:
    """Validate text accepted by log_text."""
    if not isinstance(text, str) or not text:
        raise ValueError(f"`{field_name}` must be a non-empty string")

    return text


def _normalize_logged_text(text: str, field_name: str) -> str:
    """Validate text content and convert Markdown to HTML for local rendering."""
    text = _validate_logged_text(text, field_name)

    if not is_html(text):
        return md_to_html(text, mathml=True)

    return text


def _validate_manual_log_text_args(
    text: str, prompt: Optional[str], context: Optional[Dict[str, Any]]
) -> str:
    """Validate manual log_text arguments."""
    if prompt is not None:
        raise ValueError("`prompt` is only supported when `text` is omitted")
    if context is not None:
        raise ValueError("`context` is only supported when `text` is omitted")

    return _validate_logged_text(text, "text")


def _build_log_text_generation_request(
    content_id: str,
    prompt: Optional[str],
    context: Optional[Dict[str, Any]],
    section_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the request payload for AI-assisted text generation."""
    request_data = {
        "content_id": content_id,
        "generate": True,
    }

    if prompt is not None:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("`prompt` must be a non-empty string")
        request_data["prompt"] = prompt

    if section_id is not None:
        if not isinstance(section_id, str) or not section_id:
            raise ValueError("`section_id` must be a non-empty string")
        request_data["section_id"] = section_id

    validated_context = _validate_log_text_context(context)
    if validated_context is not None:
        request_data["context"] = validated_context

    return request_data


def _generate_log_text_source(
    content_id: str,
    prompt: Optional[str],
    context: Optional[Dict[str, Any]],
    section_id: Optional[str] = None,
) -> str:
    """Generate and validate source text without converting it to HTML."""
    request_data = _build_log_text_generation_request(
        content_id,
        prompt,
        context,
        section_id=section_id,
    )
    generated_text = generate_qualitative_text(request_data)["content"]
    return _validate_logged_text(generated_text, "generated text")


def _generate_log_text(
    content_id: str,
    prompt: Optional[str],
    context: Optional[Dict[str, Any]],
    section_id: Optional[str] = None,
) -> str:
    """Generate text and normalize it to HTML for local result rendering."""
    generated_text = _generate_log_text_source(
        content_id,
        prompt,
        context,
        section_id=section_id,
    )
    return _normalize_logged_text(generated_text, "generated text")


def _render_logged_text(logged_text: Dict[str, Any]) -> str:
    """Render logged text as notebook-friendly HTML."""
    from .vm_models.html_renderer import StatefulHTMLRenderer

    return StatefulHTMLRenderer.render_accordion(
        items=[logged_text["text"]],
        titles=[f"Text Block: '{logged_text['content_id']}'"],
    )


async def alog_text(
    content_id: str,
    text: Optional[str] = None,
    prompt: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    _json: Optional[Dict[str, Any]] = None,
    section_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Async variant of ``log_text`` that logs or generates text."""
    if not content_id or not isinstance(content_id, str):
        raise ValueError("`content_id` must be a non-empty string")

    if text is not None:
        text = _validate_manual_log_text_args(text, prompt, context)
    else:
        text = _generate_log_text_source(
            content_id,
            prompt,
            context,
            section_id=section_id,
        )

    text_format = None if is_html(text) else "markdown"
    return await alog_metadata(
        content_id,
        text,
        _json,
        section_id=section_id,
        text_format=text_format,
    )


def log_text(
    content_id: str,
    text: Optional[str] = None,
    prompt: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    _json: Optional[Dict[str, Any]] = None,
    section_id: Optional[str] = None,
) -> str:
    """Logs or generates free-form text to ValidMind API.

    Args:
        content_id (str): Unique content identifier for the text.
        text (str, optional): The text to log. Markdown is sent to the backend for
            HTML and math conversion. If omitted, text is generated using the
            qualitative text generation backend.
        prompt (str, optional): Custom prompt used for AI-assisted text
            generation. Only supported when `text` is omitted.
        context (dict, optional): Context object for AI-assisted text
            generation. When omitted, the full document is used as context.
            Currently only supports `{"content_ids": [<content_id>, ...]}`.
        _json (dict, optional): Additional metadata to associate with the text. Defaults to None.
        section_id (str, optional): Section ID to append the text block to when the
            content ID does not already exist.

    Raises:
        ValueError: If arguments are invalid or use incompatible combinations.
        Exception: If the API call fails.

    Returns:
        str: HTML string containing the logged text in an accordion format.
    """
    logged_text = run_async(
        alog_text,
        content_id=content_id,
        text=text,
        prompt=prompt,
        context=context,
        _json=_json,
        section_id=section_id,
    )
    return _render_logged_text(logged_text)


def _send_metric_sync(
    key: str,
    value: Union[int, float],
    inputs: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    recorded_at: Optional[str] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    passed: Optional[bool] = None,
):
    """Send one metric without creating or depending on an event loop."""
    _ensure_fresh_oidc_token()
    try:
        return post_metric(
            _get_url("log_unit_metric"),
            serialize_metric(
                key,
                value,
                inputs,
                params,
                recorded_at,
                thresholds,
                passed,
                encoder=NumpyEncoder,
            ),
            _get_api_headers(),
            timeout=float(os.getenv("VM_API_TIMEOUT", 30)),
        )
    except TrackingAPIError as e:
        _raise_for_api_error(e.status_code, e.response_text)


async def alog_metric(
    key: str,
    value: Union[int, float],
    inputs: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    recorded_at: Optional[str] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    passed: Optional[bool] = None,
):
    """See log_metric for details, without blocking the current event loop."""
    try:
        return await asyncio.to_thread(
            _send_metric_sync,
            key,
            value,
            inputs,
            params,
            recorded_at,
            thresholds,
            passed,
        )
    except Exception as e:
        logger.error("Error logging metric to ValidMind API")
        raise e


def log_metric(
    key: str,
    value: Union[int, float],
    inputs: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    recorded_at: Optional[str] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    passed: Optional[bool] = None,
):
    """Logs a unit metric.

    Unit metrics are key-value pairs where the key is the metric name and the value is
    a scalar (int or float). These key-value pairs are associated
    with the currently selected model (inventory model in the ValidMind Platform) and keys
    can be logged to over time to create a history of the metric. On the ValidMind Platform,
    these metrics will be used to create plots/visualizations for documentation and dashboards etc.

    Args:
        key (str): The metric key
        value (Union[int, float]): The metric value (scalar)
        inputs (List[str], optional): List of input IDs
        params (Dict[str, Any], optional): Parameters used to generate the metric
        recorded_at (str, optional): Timestamp when the metric was recorded
        thresholds (Dict[str, Any], optional): Thresholds for the metric
        passed (bool, optional): Whether the metric passed validation thresholds
    """
    try:
        return _send_metric_sync(
            key,
            value,
            inputs,
            params,
            recorded_at,
            thresholds,
            passed,
        )
    except Exception as e:
        logger.error("Error logging metric to ValidMind API")
        raise e


def generate_test_result_description(test_result_data: Dict[str, Any]) -> str:
    _ensure_fresh_oidc_token()
    r = requests.post(
        url=_get_url("ai/generate/test_result_description"),
        headers=_get_api_headers(),
        json=test_result_data,
    )

    if r.status_code != 200:
        _raise_for_api_error(r.status_code, r.text)

    return r.json()
