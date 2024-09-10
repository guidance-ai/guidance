import pytest

from guidance.library._rfc3986 import uri
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
            grammar_callable=lambda name: uri(name=name),
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
