"""Integration tests verifying URI validation parameters propagate through audio/image/video."""

import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from guidance._uri_validation import URIValidationError
from guidance._utils import _ValidatingRedirectHandler, bytes_from


class TestBytesFromValidation:
    def test_blocks_http_by_default(self):
        with pytest.raises(URIValidationError, match="scheme 'http' is not in allowed schemes"):
            bytes_from("http://example.com/file.bin", allow_local=True)

    def test_allows_http_when_configured(self):
        with patch("urllib.request.build_opener") as mock_build_opener:
            mock_open = mock_build_opener.return_value.open
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = lambda s, *a: None
            mock_open.return_value.geturl.return_value = "http://example.com/file.bin"
            mock_open.return_value.read.return_value = b"data"
            with patch("guidance._uri_validation._validate_hostname_not_private"):
                bytes_from("http://example.com/file.bin", allow_local=True, allowed_schemes=("http", "https"))

    def test_blocks_private_ip_by_default(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("10.0.0.1", 443))]):
            with pytest.raises(URIValidationError, match="private/loopback/link-local/reserved"):
                bytes_from("https://internal.example.com/file.bin", allow_local=True)

    def test_allows_private_when_configured(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("10.0.0.1", 443))]):
            with patch("urllib.request.build_opener") as mock_build_opener:
                mock_open = mock_build_opener.return_value.open
                mock_open.return_value.__enter__ = lambda s: s
                mock_open.return_value.__exit__ = lambda s, *a: None
                mock_open.return_value.geturl.return_value = "https://internal.example.com/file.bin"
                mock_open.return_value.read.return_value = b"data"
                result = bytes_from("https://internal.example.com/file.bin", allow_local=True, allow_private=True)
                assert result == b"data"

    def test_blocks_file_uri_when_allow_local_false(self):
        with pytest.raises(URIValidationError, match="file:// URIs are not allowed"):
            bytes_from("file:///etc/passwd", allow_local=False)

    def test_allows_file_uri_when_allow_local_true(self, tmp_path):
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"file content")
        result = bytes_from(test_file.as_uri(), allow_local=True)
        assert result == b"file content"

    def test_allows_file_uri_without_prior_urlopen_patch(self, tmp_path):
        test_file = tmp_path / "clean.bin"
        test_file.write_bytes(b"clean file")

        assert bytes_from(test_file.as_uri(), allow_local=True) == b"clean file"

    def test_revalidates_redirect_target(self):
        def fake_getaddrinfo(hostname, *args, **kwargs):
            if hostname == "example.com":
                return [(None, None, None, None, ("93.184.216.34", 443))]
            if hostname == "internal.example.com":
                return [(None, None, None, None, ("10.0.0.1", 443))]
            raise AssertionError(f"unexpected hostname: {hostname}")

        with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
            with patch("urllib.request.build_opener") as mock_build_opener:
                response = mock_build_opener.return_value.open.return_value.__enter__.return_value
                response.geturl.return_value = "https://internal.example.com/file.bin"
                response.read.return_value = b"private data"

                with pytest.raises(URIValidationError, match="private/loopback/link-local/reserved"):
                    bytes_from("https://example.com/file.bin", allow_local=True)

                response.read.assert_not_called()

    def test_rejects_non_string_redirect_target(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 443))]):
            with patch("urllib.request.build_opener") as mock_build_opener:
                response = mock_build_opener.return_value.open.return_value.__enter__.return_value
                response.geturl.return_value = b"https://example.com/file.bin"
                response.read.return_value = b"data"

                with pytest.raises(TypeError, match="response URL to be a string"):
                    bytes_from("https://example.com/file.bin", allow_local=True)

                response.read.assert_not_called()

    def test_redirect_handler_rejects_before_following(self):
        handler = _ValidatingRedirectHandler(allowed_schemes=("https",), allow_private=False, allow_local=True)
        request = urllib.request.Request("https://example.com/start")
        with patch(
            "guidance._uri_validation.validate_uri",
            side_effect=URIValidationError("blocked redirect"),
        ) as validate_uri:
            with patch.object(urllib.request.HTTPRedirectHandler, "redirect_request") as parent:
                with pytest.raises(URIValidationError, match="blocked redirect"):
                    handler.redirect_request(
                        request,
                        None,
                        302,
                        "Found",
                        {"Location": "https://internal.example/secret"},
                        "https://internal.example/secret",
                    )
        validate_uri.assert_called_once_with(
            "https://internal.example/secret",
            allowed_schemes=("https",),
            allow_private=False,
            allow_local=True,
        )
        parent.assert_not_called()

    def test_validates_each_redirect_hop_with_real_http_server(self):
        requests_seen = []

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                requests_seen.append(self.path)
                if self.path == "/start":
                    self.send_response(302)
                    self.send_header("Location", "/middle")
                    self.end_headers()
                elif self.path == "/middle":
                    self.send_response(302)
                    self.send_header("Location", "/final")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"final payload")

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = bytes_from(
                f"http://127.0.0.1:{server.server_port}/start",
                allow_local=True,
                allowed_schemes=("http",),
                allow_private=True,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        assert result == b"final payload"
        assert requests_seen == ["/start", "/middle", "/final"]

    def test_local_path_still_works(self, tmp_path):
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"local data")
        result = bytes_from(str(test_file), allow_local=True)
        assert result == b"local data"

    def test_bytes_passthrough_unchanged(self):
        result = bytes_from(b"raw bytes", allow_local=True)
        assert result == b"raw bytes"

    def test_local_path_blocked_when_allow_local_false(self, tmp_path):
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"data")
        with pytest.raises(Exception, match="Unable to load bytes"):
            bytes_from(str(test_file), allow_local=False)


class TestMaxBytesAndTimeout:
    def test_local_file_exceeding_max_bytes_raises(self, tmp_path):
        test_file = tmp_path / "big.bin"
        test_file.write_bytes(b"x" * 100)
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            bytes_from(str(test_file), allow_local=True, max_bytes=50)

    def test_local_file_at_max_bytes_succeeds(self, tmp_path):
        test_file = tmp_path / "exact.bin"
        test_file.write_bytes(b"x" * 50)
        result = bytes_from(str(test_file), allow_local=True, max_bytes=50)
        assert result == b"x" * 50

    def test_bytes_exceeding_max_bytes_raises(self):
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            bytes_from(b"x" * 100, allow_local=True, max_bytes=50)

    def test_timeout_passed_to_urlopen(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("8.8.8.8", 443))]):
            with patch("urllib.request.build_opener") as mock_build_opener:
                mock_open = mock_build_opener.return_value.open
                mock_open.return_value.__enter__ = lambda s: s
                mock_open.return_value.__exit__ = lambda s, *a: None
                mock_open.return_value.geturl.return_value = "https://example.com/f"
                mock_open.return_value.read.return_value = b"data"
                bytes_from("https://example.com/f", allow_local=True, timeout=42.0)
                mock_open.assert_called_once_with("https://example.com/f", timeout=42.0)
