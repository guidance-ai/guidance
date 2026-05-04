"""Integration tests verifying URI validation parameters propagate through audio/image/video."""

from unittest.mock import patch

import pytest

from guidance._uri_validation import URIValidationError
from guidance._utils import bytes_from


class TestBytesFromValidation:
    def test_blocks_http_by_default(self):
        with pytest.raises(URIValidationError, match="scheme 'http' is not in allowed schemes"):
            bytes_from("http://example.com/file.bin", allow_local=True)

    def test_allows_http_when_configured(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = lambda s: s
            mock_urlopen.return_value.__exit__ = lambda s, *a: None
            mock_urlopen.return_value.read.return_value = b"data"
            with patch("guidance._uri_validation._validate_hostname_not_private"):
                bytes_from("http://example.com/file.bin", allow_local=True, allowed_schemes=("http", "https"))

    def test_blocks_private_ip_by_default(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("10.0.0.1", 443))]):
            with pytest.raises(URIValidationError, match="private/loopback/link-local/reserved"):
                bytes_from("https://internal.example.com/file.bin", allow_local=True)

    def test_allows_private_when_configured(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("10.0.0.1", 443))]):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__ = lambda s: s
                mock_urlopen.return_value.__exit__ = lambda s, *a: None
                mock_urlopen.return_value.read.return_value = b"data"
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
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__ = lambda s: s
                mock_urlopen.return_value.__exit__ = lambda s, *a: None
                mock_urlopen.return_value.read.return_value = b"data"
                bytes_from("https://example.com/f", allow_local=True, timeout=42.0)
                mock_urlopen.assert_called_once_with("https://example.com/f", timeout=42.0)
