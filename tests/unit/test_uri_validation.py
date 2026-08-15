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
            "169.254.1.1",
            # 169.254.169.254 lives in this range too, but it is now rejected by the metadata
            # check first and reports a different message. See TestCloudMetadataEndpoints.
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


METADATA_ADDRESSES = [
    "169.254.169.254",  # AWS IMDS, GCP, Azure, OCI, DigitalOcean
    "169.254.170.2",  # AWS ECS task IAM role credentials
    "169.254.170.23",  # AWS EKS Pod Identity Agent
    "168.63.129.16",  # Azure WireServer (publicly routable)
    "100.100.100.200",  # Alibaba Cloud (100.64.0.0/10, not reported as private)
    "192.0.0.192",  # Oracle Cloud (Classic)
    "169.254.42.42",  # Scaleway
]


class TestCloudMetadataEndpoints:
    @pytest.mark.parametrize("ip", METADATA_ADDRESSES)
    def test_metadata_endpoints_blocked_by_default(self, ip):
        with pytest.raises(URIValidationError):
            validate_uri(f"https://{ip}/latest/meta-data/")

    @pytest.mark.parametrize("ip", METADATA_ADDRESSES)
    def test_metadata_endpoints_blocked_even_with_allow_private(self, ip):
        """`allow_private` is for reaching your own network, not the credential endpoint."""
        with pytest.raises(URIValidationError, match="cloud metadata endpoint"):
            validate_uri(f"https://{ip}/latest/meta-data/", allow_private=True)

    def test_allow_private_still_permits_rfc1918(self):
        validate_uri("https://10.0.0.5/resource", allow_private=True)

    def test_allow_private_still_permits_loopback(self):
        validate_uri("https://127.0.0.1/resource", allow_private=True)

    def test_public_address_still_allowed(self):
        validate_uri("https://1.1.1.1/resource")

    def test_metadata_endpoint_via_dns_blocked(self):
        """The check runs on every resolved address, not just on IP literals."""
        addrinfo = [(2, 1, 6, "", ("168.63.129.16", 0))]
        with (
            patch("guidance._uri_validation.socket.getaddrinfo", return_value=addrinfo),
            pytest.raises(URIValidationError, match="cloud metadata endpoint"),
        ):
            validate_uri("https://metadata.example.com/x")

    @pytest.mark.parametrize(
        "ip",
        [
            "168.63.129.16",  # Azure WireServer, publicly routable
            "100.100.100.200",  # Alibaba Cloud, inside 100.64.0.0/10 which Python calls public
        ],
    )
    def test_metadata_endpoint_not_reachable_through_ipv4_mapped_ipv6(self, ip):
        """Wrapping the address in `::ffff:` must not walk past the list.

        These are the two entries that motivated the list: both are publicly routable, so the
        private/loopback/link-local/reserved checks never catch them and the membership test is
        the only guard. The link-local entries hide this, because Python reports an IPv4-mapped
        link-local address as link-local and blocks it on the range check regardless.
        """
        with pytest.raises(URIValidationError, match="cloud metadata endpoint"):
            validate_uri(f"https://[::ffff:{ip}]/latest/meta-data/")
