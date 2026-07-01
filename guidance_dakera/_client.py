"""
Low-level HTTP client for the Dakera memory server.

Deliberately dependency-light: only `urllib` from the stdlib is required.
If `requests` is installed it is used for a slightly nicer experience, but
it is never mandatory.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DakeraClient:
    """Minimal HTTP client for the Dakera v1 REST API.

    Parameters
    ----------
    base_url:
        Base URL of the Dakera server, e.g. ``"http://localhost:3300"``.
        Defaults to the ``DAKERA_BASE_URL`` environment variable, then
        ``"http://localhost:3300"``.
    api_key:
        Bearer token sent in the ``Authorization`` header.
        Defaults to ``DAKERA_API_KEY``.  Pass ``None`` to skip auth (works
        when the server is started without ``DAKERA_API_KEY`` set).
    timeout:
        Socket timeout in seconds for every request (default 10).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("DAKERA_BASE_URL", "http://localhost:3300")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("DAKERA_API_KEY")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise DakeraAPIError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise DakeraConnectionError(str(exc.reason)) from exc

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        agent_id: str,
        *,
        session_id: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Store a memory and return the created memory record."""
        payload: dict[str, Any] = {"content": content, "agent_id": agent_id}
        if session_id:
            payload["session_id"] = session_id
        if importance is not None:
            payload["importance"] = importance
        if tags:
            payload["tags"] = tags
        return self._request("POST", "/v1/memory/store", payload)

    def search(
        self,
        query: str,
        agent_id: str,
        *,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search memories; returns a list of ``{memory, score}`` dicts."""
        payload: dict[str, Any] = {
            "query": query,
            "agent_id": agent_id,
            "top_k": top_k,
        }
        if session_id:
            payload["session_id"] = session_id
        result = self._request("POST", "/v1/memory/search", payload)
        return result.get("memories", [])

    def forget(
        self,
        agent_id: str,
        memory_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Forget specific memories (or all memories for an agent)."""
        payload: dict[str, Any] = {"agent_id": agent_id}
        if memory_ids:
            payload["memory_ids"] = memory_ids
        return self._request("POST", "/v1/memory/forget", payload)

    def health(self) -> dict[str, Any]:
        """Return the server health/readiness payload."""
        return self._request("GET", "/health")


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------

class DakeraError(RuntimeError):
    """Base class for all Dakera client errors."""


class DakeraAPIError(DakeraError):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Dakera API error {status_code}: {detail}")


class DakeraConnectionError(DakeraError):
    """Raised when the server cannot be reached."""
