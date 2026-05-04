"""Tests for URI validation security controls."""

from unittest.mock import patch

import pytest

from guidance._uri_validation import URIValidationError, validate_uri


class TestSchemeAllowlist:
    def test_https_allowed_by_default(self):
        # Should not raise for https with a public IP
        with patch("guidance._uri_validation._validate_hostname_not_private"):
            validate_uri("https://example.com/resource")

    def test_http_blocked_by_default(self):
        with pytest.raises(URIValidationError, match="scheme 'http' is not in allowed schemes"):
            validate_uri("http://example.com/resource")

    def test_ftp_blocked_by_default(self):
        with pytest.raises(URIValidationError, match="scheme 'ftp' is not in allowed schemes"):
            validate_uri("ftp://example.com/resource")

    def test_http_allowed_when_configured(self):
        with patch("guidance._uri_validation._validate_hostname_not_private"):
            validate_uri("http://example.com/resource", allowed_schemes=("http", "https"))

    def test_custom_scheme_allowed(self):
        with patch("guidance._uri_validation._validate_hostname_not_private"):
            validate_uri("s3://bucket/key", allowed_schemes=("s3",))

    def test_scheme_check_is_case_insensitive(self):
        with patch("guidance._uri_validation._validate_hostname_not_private"):
            validate_uri("HTTPS://example.com/resource", allowed_schemes=("https",))


class TestFileURIBlocking:
    def test_file_uri_blocked_when_allow_local_false(self):
        with pytest.raises(URIValidationError, match="file:// URIs are not allowed"):
            validate_uri("file:///etc/passwd", allow_local=False)

    def test_file_uri_allowed_when_allow_local_true(self):
        # Should not raise
        validate_uri("file:///some/path", allow_local=True)

    def test_file_uri_not_subject_to_scheme_allowlist(self):
        # file:// is governed by allow_local, not the scheme allowlist
        validate_uri("file:///some/path", allowed_schemes=("https",), allow_local=True)


class TestPrivateIPBlocking:
    @pytest.mark.parametrize(
        "ip",
        [
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.255.255",
        ],
    )
    def test_rfc1918_private_ranges_blocked(self, ip):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, (ip, 443))]):
            with pytest.raises(URIValidationError, match="private/loopback/link-local/reserved"):
                validate_uri(f"https://{ip}/resource")

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "127.0.0.2",
            "127.255.255.255",
        ],
    )
    def test_loopback_blocked(self, ip):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, (ip, 443))]):
            with pytest.raises(URIValidationError, match="private/loopback/link-local/reserved"):
                validate_uri("https://localhost/resource")

    @pytest.mark.parametrize(
        "ip",
        [
            "169.254.0.1",
            "169.254.169.254",  # AWS metadata endpoint
        ],
    )
    def test_link_local_blocked(self, ip):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, (ip, 443))]):
            with pytest.raises(URIValidationError, match="private/loopback/link-local/reserved"):
                validate_uri("https://metadata.internal/resource")

    @pytest.mark.parametrize(
        "ip",
        [
            "::1",  # IPv6 loopback
            "fe80::1",  # IPv6 link-local
        ],
    )
    def test_ipv6_private_blocked(self, ip):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, (ip, 443, 0, 0))]):
            with pytest.raises(URIValidationError, match="private/loopback/link-local/reserved"):
                validate_uri(f"https://[{ip}]/resource")

    def test_ip_literal_in_url_blocked(self):
        with pytest.raises(URIValidationError, match="private/loopback/link-local/reserved"):
            validate_uri("https://127.0.0.1/resource")

    def test_allow_private_permits_rfc1918(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("10.0.0.1", 443))]):
            # Should not raise
            validate_uri("https://internal.corp/resource", allow_private=True)

    def test_public_ip_allowed(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("8.8.8.8", 443))]):
            validate_uri("https://example.com/resource")

    def test_dns_resolution_failure_raises(self):
        import socket as _socket

        with patch("socket.getaddrinfo", side_effect=_socket.gaierror("Name resolution failed")):
            with pytest.raises(URIValidationError, match="Failed to resolve hostname"):
                validate_uri("https://nonexistent.invalid/resource")

    def test_no_hostname_raises(self):
        with pytest.raises(URIValidationError, match="URI has no hostname"):
            validate_uri("https:///path/only")

    def test_all_resolved_addresses_checked(self):
        # If DNS returns multiple IPs, all must be validated
        addrinfos = [
            (None, None, None, None, ("8.8.8.8", 443)),
            (None, None, None, None, ("10.0.0.1", 443)),  # private!
        ]
        with patch("socket.getaddrinfo", return_value=addrinfos):
            with pytest.raises(URIValidationError, match="private/loopback/link-local/reserved"):
                validate_uri("https://tricky.example.com/resource")
