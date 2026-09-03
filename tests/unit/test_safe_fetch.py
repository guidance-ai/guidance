"""Tests for SSRF hardening in fetch_url_bytes / bytes_from."""

from unittest.mock import MagicMock, patch

import pytest

from guidance._safe_fetch import fetch_url_bytes
from guidance._uri_validation import URIValidationError
from guidance._utils import bytes_from


class TestRedirectRevalidation:
    def test_redirect_to_private_ip_is_blocked(self):
        mock_resp = MagicMock()
        mock_resp.status = 302
        mock_resp.getheaders.return_value = [("Location", "https://169.254.169.254/latest/meta-data/")]
        mock_resp.read.return_value = b""

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        with (
            patch(
                "guidance._uri_validation.socket.getaddrinfo",
                return_value=[(None, None, None, None, ("8.8.8.8", 443))],
            ),
            patch(
                "guidance._safe_fetch.socket.getaddrinfo",
                return_value=[(None, None, None, None, ("8.8.8.8", 443))],
            ),
            patch("guidance._safe_fetch._PinnedHTTPSConnection", return_value=mock_conn),
        ):
            # Redirect target is a link-local IP literal → blocked by validate_uri.
            with pytest.raises(URIValidationError, match="private/loopback/link-local/reserved"):
                fetch_url_bytes(
                    "https://example.com/start",
                    allowed_schemes=("https",),
                    allow_private=False,
                    allow_local=False,
                    max_bytes=1024,
                    timeout=5,
                )


class TestDnsPinning:
    def test_connects_to_resolved_ip_not_hostname(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.getheaders.return_value = []
        mock_resp.read.return_value = b"ok"

        created: dict[str, str] = {}

        def fake_conn(hostname, pinned_ip, port, timeout, context):
            created["hostname"] = hostname
            created["pinned_ip"] = pinned_ip
            conn = MagicMock()
            conn.getresponse.return_value = mock_resp
            return conn

        with (
            patch(
                "guidance._uri_validation.socket.getaddrinfo",
                return_value=[(None, None, None, None, ("93.184.216.34", 443))],
            ),
            patch(
                "guidance._safe_fetch.socket.getaddrinfo",
                return_value=[(None, None, None, None, ("93.184.216.34", 443))],
            ),
            patch("guidance._safe_fetch._PinnedHTTPSConnection", side_effect=fake_conn),
        ):
            data = fetch_url_bytes(
                "https://example.com/resource",
                allowed_schemes=("https",),
                allow_private=False,
                allow_local=False,
                max_bytes=1024,
                timeout=5,
            )

        assert data == b"ok"
        assert created["hostname"] == "example.com"
        assert created["pinned_ip"] == "93.184.216.34"


class TestBytesFromUsesSafeFetch:
    def test_http_path_uses_fetch_url_bytes(self):
        with patch("guidance._safe_fetch.fetch_url_bytes", return_value=b"payload") as mocked:
            result = bytes_from(
                "https://example.com/x",
                allow_local=False,
                allowed_schemes=("https",),
                allow_private=True,
            )
            assert result == b"payload"
            mocked.assert_called_once()
