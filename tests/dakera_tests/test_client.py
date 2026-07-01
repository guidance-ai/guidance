"""
Unit tests for DakeraClient.

These tests mock the HTTP layer so no live Dakera server is needed.
"""

from __future__ import annotations

import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from guidance_dakera._client import DakeraAPIError, DakeraClient, DakeraConnectionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(payload: dict, status: int = 200):
    """Return a mock urllib response object."""
    body = json.dumps(payload).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.status = status
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _patch_urlopen(payload: dict, status: int = 200):
    return patch(
        "urllib.request.urlopen",
        return_value=_make_response(payload, status),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDakeraClientInit(unittest.TestCase):
    def test_defaults_from_env(self):
        with patch.dict("os.environ", {"DAKERA_BASE_URL": "http://host:9999", "DAKERA_API_KEY": "tok"}):
            c = DakeraClient()
        self.assertEqual(c.base_url, "http://host:9999")
        self.assertEqual(c.api_key, "tok")

    def test_explicit_args_override_env(self):
        with patch.dict("os.environ", {"DAKERA_BASE_URL": "http://env:3300"}):
            c = DakeraClient(base_url="http://explicit:1234", api_key="mykey")
        self.assertEqual(c.base_url, "http://explicit:1234")
        self.assertEqual(c.api_key, "mykey")

    def test_trailing_slash_stripped(self):
        c = DakeraClient(base_url="http://localhost:3300/")
        self.assertEqual(c.base_url, "http://localhost:3300")

    def test_no_auth_header_when_no_key(self):
        # Explicitly clear any env key so this test is hermetic
        with patch.dict("os.environ", {}, clear=True):
            c = DakeraClient(base_url="http://localhost:3300", api_key=None)
        headers = c._headers()
        self.assertNotIn("Authorization", headers)

    def test_auth_header_present_with_key(self):
        c = DakeraClient(base_url="http://localhost:3300", api_key="secret")
        self.assertEqual(c._headers()["Authorization"], "Bearer secret")


class TestDakeraClientStore(unittest.TestCase):
    def _client(self):
        return DakeraClient(base_url="http://localhost:3300", api_key="test")

    def test_store_sends_correct_payload(self):
        response_payload = {"memory": {"id": "abc123", "content": "hello"}}
        with _patch_urlopen(response_payload) as mock_open:
            result = self._client().store("hello", "agent1")

        # Inspect the request that was made
        call_args = mock_open.call_args
        req = call_args[0][0]
        sent = json.loads(req.data)
        self.assertEqual(sent["content"], "hello")
        self.assertEqual(sent["agent_id"], "agent1")
        self.assertNotIn("session_id", sent)

    def test_store_with_optional_fields(self):
        response_payload = {"memory": {"id": "xyz"}}
        with _patch_urlopen(response_payload) as mock_open:
            result = self._client().store(
                "remember this",
                "agent2",
                session_id="s1",
                importance=0.9,
                tags=["important"],
            )

        req = mock_open.call_args[0][0]
        sent = json.loads(req.data)
        self.assertEqual(sent["session_id"], "s1")
        self.assertEqual(sent["importance"], 0.9)
        self.assertEqual(sent["tags"], ["important"])

    def test_store_returns_dict(self):
        response_payload = {"memory": {"id": "m1", "content": "foo"}}
        with _patch_urlopen(response_payload):
            result = self._client().store("foo", "a")
        self.assertIn("memory", result)
        self.assertEqual(result["memory"]["id"], "m1")


class TestDakeraClientSearch(unittest.TestCase):
    def _client(self):
        return DakeraClient(base_url="http://localhost:3300")

    def test_search_returns_memories_list(self):
        payload = {
            "memories": [
                {"memory": {"id": "1", "content": "foo"}, "score": 0.95},
                {"memory": {"id": "2", "content": "bar"}, "score": 0.7},
            ]
        }
        with _patch_urlopen(payload):
            hits = self._client().search("query", "agent1")
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["score"], 0.95)

    def test_search_sends_top_k(self):
        payload = {"memories": []}
        with _patch_urlopen(payload) as mock_open:
            self._client().search("q", "a", top_k=10)
        req = mock_open.call_args[0][0]
        sent = json.loads(req.data)
        self.assertEqual(sent["top_k"], 10)

    def test_search_empty_result(self):
        with _patch_urlopen({"memories": []}):
            hits = self._client().search("nothing", "agent")
        self.assertEqual(hits, [])


class TestDakeraClientForget(unittest.TestCase):
    def _client(self):
        return DakeraClient(base_url="http://localhost:3300")

    def test_forget_all(self):
        with _patch_urlopen({}) as mock_open:
            self._client().forget("agent1")
        req = mock_open.call_args[0][0]
        sent = json.loads(req.data)
        self.assertNotIn("memory_ids", sent)

    def test_forget_specific_ids(self):
        with _patch_urlopen({}) as mock_open:
            self._client().forget("agent1", memory_ids=["id1", "id2"])
        req = mock_open.call_args[0][0]
        sent = json.loads(req.data)
        self.assertEqual(sent["memory_ids"], ["id1", "id2"])


class TestDakeraClientErrors(unittest.TestCase):
    def _client(self):
        return DakeraClient(base_url="http://localhost:3300")

    def test_api_error_raised_on_4xx(self):
        import urllib.error
        err = urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found",
            hdrs=None, fp=BytesIO(b'{"error":"not found"}')
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(DakeraAPIError) as ctx:
                self._client().store("x", "a")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_connection_error_raised(self):
        import urllib.error
        err = urllib.error.URLError("connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(DakeraConnectionError):
                self._client().health()
