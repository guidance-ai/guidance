import pytest

from guidance.library._rfc3986 import uri, ipv6address
from ...utils import generate_and_check, check_match_failure

class TestURI:
    @pytest.mark.parametrize(
        "target_obj",
        [
            "ftp://ftp.is.co.za/rfc/rfc1808.txt",
            "http://www.ietf.org/rfc/rfc2396.txt",
            "ldap://[2001:db8::7]/c=GB?objectClass?one",
            "mailto:John.Doe@example.com",
            "news:comp.infosystems.www.servers.unix",
            "tel:+1-816-555-1212",
            "telnet://192.0.2.16:80/",
            "urn:oasis:names:specification:docbook:dtd:xml:4.1.2"
        ]
    )
    def test_uri(self, target_obj):
        generate_and_check(
            grammar_callable=uri,
            test_string=target_obj,
        )

    @pytest.mark.parametrize(
        "bad_obj",
        [
            "http://example.com:-80",  # Negative port numbers are not valid.
            "telnet://example.com:abc",  # Port must be numeric, but it's a string.
            "ftp://user@:example.com",  # Missing password after the "@" in userinfo.
            "ftp://user:pass@:example.com",  # Multiple '@' characters in the userinfo section.
            "http://[::1]:80:80",  # Multiple ports specified.
            "http://[2001:db8:::7]",  # Too many colons in IPv6 address (invalid IPv6 syntax).
            "http://example.com/<>",  # Invalid characters '<' and '>' in the path.
            "http://example.com#fragment#another",  # Multiple fragment identifiers.
            "ftp://example.com/%",  # Percent symbol must be part of a valid percent-encoding.
            "http://example.com/|path",  # Pipe '|' is not allowed in the path.
            "ftp://user:password@ftp.[example].com",  # Brackets in domain name.
            "http://example.com//{}path",  # Curly braces are not allowed in the path.
            "ftp://example.com:%20",  # Invalid use of percent-encoding (incomplete sequence).
            "http://example.com/|file",  # Pipe '|' is not valid in the path.
            "http://example.com:port",  # Port must be numeric, not a string.
            "ftp://[2001:db8::1]::21",  # Multiple colons in the port specification.
            "ftp://user:password@:host.com",  # Invalid syntax with multiple '@' characters.
            "http://example.com/%zz",  # Invalid percent-encoding (invalid hex digits).
            "ftp://example.com/file%gg",  # Invalid percent-encoding (invalid hex digits).
            "ftp://user:password@example.com::21",  # Multiple port numbers specified.
            "http://example.com/\"path\"",  # Quotes are not allowed in URI paths.
            "mailto:user@ex ample.com",  # Space is not allowed in the domain.
        ]
    )
    def test_bad_uri(self, bad_obj):
        check_match_failure(
            bad_string=bad_obj,
            grammar=uri(),
        )

class TestIPv6:
    @pytest.mark.parametrize(
        "good_obj",
        [
            "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
            "2001:db8:85a3:0:0:8a2e:370:7334",  # Leading zeros omitted.
            "2001:db8:85a3::8a2e:370:7334",  # Use of "::" to compress consecutive zeros.
            "::1",  # Loopback address.
            "fe80::1ff:fe23:4567:890a",  # Link-local address.
            "2001:db8::",  # Compressed IPv6 address with all zeros after "db8".
            "::",  # Unspecified address (all zeros).
            "2001:0db8:0000:0042:0000:8a2e:0370:7334",  # Full format with leading zeros.
            "2001:db8::42:8a2e:370:7334",  # Mixed zeros compressed in the middle.
            "ff02::1",  # Multicast address for all nodes.
            "2001:db8:0:0:0:0:2:1",  # Partially compressed address.
            "2001:db8::2:1",  # More compression of zeros.
            "2001:0db8:1234::1",  # Partially compressed address with a specific host.
            "2001:db8:0:1:1:1:1:1",  # A valid IPv6 address with a mix of segments.
            "fe80::200:5aee:feaa:20a2",  # Another link-local address.
            "2001:db8:abcd:0012::0",  # Compressed address with multiple zeros.
            "2001:db8::abcd:12",  # Compress zeros at the end.
            "2001:0db8:0000:0042:abcd:8a2e:0370:7334",  # Leading zeros included.
            "ff02::2",  # Multicast address for all routers.
            "2001:0db8:85a3:0000:0000:8a2e:0370:1234",  # Similar to the first, different last segment.
        ]
    )
    def test_good_ipv6(self, good_obj):
        generate_and_check(
            grammar_callable=ipv6address,
            test_string=good_obj,
        )

    @pytest.mark.parametrize(
        "bad_obj",
        [
            "2001:db8:85a3:0:0:8a2e:370g:7334",  # Invalid hex character 'g'.
            "2001::85a3::7334",  # Multiple "::" compression in one address is not allowed.
            "2001:db8:85a3:0000:0000:8a2e:0370",  # Too few segments (only 7).
            "2001:db8:85a3:0000:0000:8a2e:0370:7334:1234",  # Too many segments (9).
            "2001:db8:85a3:0000:0000:8a2e:0370:7334::",  # "::" used at the end of a complete address.
            "12001:db8:85a3:0000:0000:8a2e:0370:7334",  # Invalid segment length (5 hex digits in the first part).
            "2001:db8:85a3:0:0:8a2e:370::7334",  # "::" in the middle but results in too many segments.
            "2001:db8:85a3:0000:0000:8a2e:0370:zzzz",  # Invalid hex character 'z'.
            "2001:db8:85a3:0000:0000:8a2e:0370:",  # Trailing colon with incomplete segment.
            "2001:db8:85a3:0000:0000:8a2e:0370:7334/64",  # CIDR notation not allowed in pure IPv6 address.
            "2001:db8:85a3:0000:0000:8a2e:03707334",  # Missing colon between segments.
            ":2001:db8:85a3:0000:0000:8a2e:0370:7334",  # Leading colon without "::".
            "2001:db8:85a3:0000:0000:0370:7334",  # Too few segments (missing 8th segment).
            "fe80::200:5aee:feaa:20a21",  # Invalid segment length (5 hex digits).
            "2001:db8:1234:5678:9abc:defg:1234:5678",  # Invalid hex character 'g'.
            "2001:db8:85a3:0000:0000:8a2e:03707334:",  # Trailing colon.
            "2001-db8-85a3-0000-0000-8a2e-0370-7334",  # Hyphens used instead of colons.
            "2001:db8:85a3:0:0:8a2e:370:7334:7334",  # Too many segments (9).
            "2001:db8:85a3:0:0:8a2e:370:7334:xyz",  # Invalid character 'xyz'.
        ]
    )
    def test_bad_ipv6(self, bad_obj):
        check_match_failure(
            bad_string=bad_obj,
            grammar=ipv6address(),
        )
