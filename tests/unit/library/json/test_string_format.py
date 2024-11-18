"""Adapted from https://github.com/json-schema-org/JSON-Schema-Test-Suite/tree/9fc880bfb6d8ccd093bc82431f17d13681ffae8e/tests/draft2020-12/optional/format"""

import json

import pytest

from .utils import check_match_failure, generate_and_check


class TestDate:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"date"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"1963-06-19"',  # a valid date string
            '"2020-01-31"',  # a valid date string with 31 days in January
            '"2021-02-28"',  # a valid date string with 28 days in February (normal)
            '"2020-02-29"',  # a valid date string with 29 days in February (leap)
            '"2020-03-31"',  # a valid date string with 31 days in March
            '"2020-04-30"',  # a valid date string with 30 days in April
            '"2020-05-31"',  # a valid date string with 31 days in May
            '"2020-06-30"',  # a valid date string with 30 days in June
            '"2020-07-31"',  # a valid date string with 31 days in July
            '"2020-08-31"',  # a valid date string with 31 days in August
            '"2020-09-30"',  # a valid date string with 30 days in September
            '"2020-10-31"',  # a valid date string with 31 days in October
            '"2020-11-30"',  # a valid date string with 30 days in November
            '"2020-12-31"',  # a valid date string with 31 days in December
            '"2020-02-29"',  # 2020 is a leap year
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"2020-01-32"',  # a invalid date string with 32 days in January
            pytest.param(
                '"2021-02-29"',
                marks=pytest.mark.xfail(reason="number of days not yet tied to month"),
            ),  # a invalid date string with 29 days in February (normal)
            pytest.param(
                '"2020-02-30"',
                marks=pytest.mark.xfail(reason="number of days not yet tied to month"),
            ),  # a invalid date string with 30 days in February (leap)
            '"2020-03-32"',  # a invalid date string with 32 days in March
            pytest.param(
                '"2020-04-31"',
                marks=pytest.mark.xfail(reason="number of days not yet tied to month"),
            ),  # a invalid date string with 31 days in April
            '"2020-05-32"',  # a invalid date string with 32 days in May
            pytest.param(
                '"2020-06-31"',
                marks=pytest.mark.xfail(reason="number of days not yet tied to month"),
            ),  # a invalid date string with 31 days in June
            '"2020-07-32"',  # a invalid date string with 32 days in July
            '"2020-08-32"',  # a invalid date string with 32 days in August
            pytest.param(
                '"2020-09-31"',
                marks=pytest.mark.xfail(reason="number of days not yet tied to month"),
            ),  # a invalid date string with 31 days in September
            '"2020-10-32"',  # a invalid date string with 32 days in October
            pytest.param(
                '"2020-11-31"',
                marks=pytest.mark.xfail(reason="number of days not yet tied to month"),
            ),  # a invalid date string with 31 days in November
            '"2020-12-32"',  # a invalid date string with 32 days in December
            '"2020-13-01"',  # a invalid date string with invalid month
            '"06/19/1963"',  # an invalid date string
            '"2013-350"',  # only RFC3339 not all of ISO 8601 are valid
            '"1998-1-20"',  # non-padded month dates are not valid
            '"1998-01-1"',  # non-padded day dates are not valid
            '"1998-13-01"',  # invalid month
            pytest.param(
                '"1998-04-31"',
                marks=pytest.mark.xfail(reason="number of days not yet tied to month"),
            ),  # invalid month-day combination
            pytest.param(
                '"2021-02-29"', marks=pytest.mark.xfail(reason="leap days are hard")
            ),  # 2021 is not a leap year
            '"1963-06-1\\u09ea"',  # invalid non-ASCII '৪' (a Bengali 4)
            '"20230328"',  # ISO8601 / non-RFC3339: YYYYMMDD without dashes (2023-03-28)
            '"2023-W01"',  # ISO8601 / non-RFC3339: week number implicit day of week (2023-01-02)
            '"2023-W13-2"',  # ISO8601 / non-RFC3339: week number with day of week (2023-03-28)
            '"2022W527"',  # ISO8601 / non-RFC3339: week number rollover to next year (2023-01-01)
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


@pytest.mark.xfail(reason="json-pointer format not implemented")
class TestJsonPointer:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"json-pointer"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"/foo/bar~0/baz~1/%a"',  # a valid JSON-pointer
            '"/foo//bar"',  # valid JSON-pointer with empty segment
            '"/foo/bar/"',  # valid JSON-pointer with the last empty segment
            '""',  # valid JSON-pointer as stated in RFC 6901 #1
            '"/foo"',  # valid JSON-pointer as stated in RFC 6901 #2
            '"/foo/0"',  # valid JSON-pointer as stated in RFC 6901 #3
            '"/"',  # valid JSON-pointer as stated in RFC 6901 #4
            '"/a~1b"',  # valid JSON-pointer as stated in RFC 6901 #5
            '"/c%d"',  # valid JSON-pointer as stated in RFC 6901 #6
            '"/e^f"',  # valid JSON-pointer as stated in RFC 6901 #7
            '"/g|h"',  # valid JSON-pointer as stated in RFC 6901 #8
            '"/i\\\\j"',  # valid JSON-pointer as stated in RFC 6901 #9
            '"/k\\"l"',  # valid JSON-pointer as stated in RFC 6901 #10
            '"/ "',  # valid JSON-pointer as stated in RFC 6901 #11
            '"/m~0n"',  # valid JSON-pointer as stated in RFC 6901 #12
            '"/foo/-"',  # valid JSON-pointer used adding to the last array position
            '"/foo/-/bar"',  # valid JSON-pointer (- used as object member name)
            '"/~1~0~0~1~1"',  # valid JSON-pointer (multiple escaped characters)
            '"/~1.1"',  # valid JSON-pointer (escaped with fraction part) #1
            '"/~0.1"',  # valid JSON-pointer (escaped with fraction part) #2
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"/foo/bar~"',  # not a valid JSON-pointer (~ not escaped)
            '"#"',  # not a valid JSON-pointer (URI Fragment Identifier) #1
            '"#/"',  # not a valid JSON-pointer (URI Fragment Identifier) #2
            '"#a"',  # not a valid JSON-pointer (URI Fragment Identifier) #3
            '"/~0~"',  # not a valid JSON-pointer (some escaped, but not all) #1
            '"/~0/~"',  # not a valid JSON-pointer (some escaped, but not all) #2
            '"/~2"',  # not a valid JSON-pointer (wrong escape character) #1
            '"/~-1"',  # not a valid JSON-pointer (wrong escape character) #2
            '"/~~"',  # not a valid JSON-pointer (multiple characters not escaped)
            '"a"',  # not a valid JSON-pointer (isn't empty nor starts with /) #1
            '"0"',  # not a valid JSON-pointer (isn't empty nor starts with /) #2
            '"a/a"',  # not a valid JSON-pointer (isn't empty nor starts with /) #3
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


@pytest.mark.xfail(reason="idn-hostname format not implemented")
class TestIdnHostname:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"idn-hostname"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"\\uc2e4\\ub840.\\ud14c\\uc2a4\\ud2b8"',  # a valid host name (example.test in Hangul)
            '"xn--ihqwcrb4cv8a8dqg056pqjye"',  # valid Chinese Punycode
            '"\\u00df\\u03c2\\u0f0b\\u3007"',  # Exceptions that are PVALID, left-to-right chars
            '"\\u06fd\\u06fe"',  # Exceptions that are PVALID, right-to-left chars
            '"l\\u00b7l"',  # MIDDLE DOT with surrounding 'l's
            '"\\u03b1\\u0375\\u03b2"',  # Greek KERAIA followed by Greek
            '"\\u05d0\\u05f3\\u05d1"',  # Hebrew GERESH preceded by Hebrew
            '"\\u05d0\\u05f4\\u05d1"',  # Hebrew GERSHAYIM preceded by Hebrew
            '"\\u30fb\\u3041"',  # KATAKANA MIDDLE DOT with Hiragana
            '"\\u30fb\\u30a1"',  # KATAKANA MIDDLE DOT with Katakana
            '"\\u30fb\\u4e08"',  # KATAKANA MIDDLE DOT with Han
            '"\\u0628\\u0660\\u0628"',  # Arabic-Indic digits not mixed with Extended Arabic-Indic digits
            '"\\u06f00"',  # Extended Arabic-Indic digits not mixed with Arabic-Indic digits
            '"\\u0915\\u094d\\u200d\\u0937"',  # ZERO WIDTH JOINER preceded by Virama
            '"\\u0915\\u094d\\u200c\\u0937"',  # ZERO WIDTH NON-JOINER preceded by Virama
            '"\\u0628\\u064a\\u200c\\u0628\\u064a"',  # ZERO WIDTH NON-JOINER not preceded by Virama but matches regexp
            '"hostname"',  # single label
            '"host-name"',  # single label with hyphen
            '"h0stn4me"',  # single label with digits
            '"1host"',  # single label starting with digit
            '"hostnam3"',  # single label ending with digit
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"\\u302e\\uc2e4\\ub840.\\ud14c\\uc2a4\\ud2b8"',  # illegal first char U+302E Hangul single dot tone mark
            '"\\uc2e4\\u302e\\ub840.\\ud14c\\uc2a4\\ud2b8"',  # contains illegal char U+302E Hangul single dot tone mark
            '"\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\uc2e4\\ub840\\ub840\\ud14c\\uc2a4\\ud2b8\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ud14c\\uc2a4\\ud2b8\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ud14c\\uc2a4\\ud2b8\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ub840\\ud14c\\uc2a4\\ud2b8\\ub840\\ub840\\uc2e4\\ub840.\\ud14c\\uc2a4\\ud2b8"',  # a host name with a component too long
            '"-> $1.00 <--"',  # invalid label, correct Punycode
            '"xn--X"',  # invalid Punycode
            '"XN--aa---o47jg78q"',  # U-label contains "--" in the 3rd and 4th position
            '"-hello"',  # U-label starts with a dash
            '"hello-"',  # U-label ends with a dash
            '"-hello-"',  # U-label starts and ends with a dash
            '"\\u0903hello"',  # Begins with a Spacing Combining Mark
            '"\\u0300hello"',  # Begins with a Nonspacing Mark
            '"\\u0488hello"',  # Begins with an Enclosing Mark
            '"\\u0640\\u07fa"',  # Exceptions that are DISALLOWED, right-to-left chars
            '"\\u3031\\u3032\\u3033\\u3034\\u3035\\u302e\\u302f\\u303b"',  # Exceptions that are DISALLOWED, left-to-right chars
            '"a\\u00b7l"',  # MIDDLE DOT with no preceding 'l'
            '"\\u00b7l"',  # MIDDLE DOT with nothing preceding
            '"l\\u00b7a"',  # MIDDLE DOT with no following 'l'
            '"l\\u00b7"',  # MIDDLE DOT with nothing following
            '"\\u03b1\\u0375S"',  # Greek KERAIA not followed by Greek
            '"\\u03b1\\u0375"',  # Greek KERAIA not followed by anything
            '"A\\u05f3\\u05d1"',  # Hebrew GERESH not preceded by Hebrew
            '"\\u05f3\\u05d1"',  # Hebrew GERESH not preceded by anything
            '"A\\u05f4\\u05d1"',  # Hebrew GERSHAYIM not preceded by Hebrew
            '"\\u05f4\\u05d1"',  # Hebrew GERSHAYIM not preceded by anything
            '"def\\u30fbabc"',  # KATAKANA MIDDLE DOT with no Hiragana, Katakana, or Han
            '"\\u30fb"',  # KATAKANA MIDDLE DOT with no other characters
            '"\\u0628\\u0660\\u06f0"',  # Arabic-Indic digits mixed with Extended Arabic-Indic digits
            '"\\u0915\\u200d\\u0937"',  # ZERO WIDTH JOINER not preceded by Virama
            '"\\u200d\\u0937"',  # ZERO WIDTH JOINER not preceded by anything
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


@pytest.mark.xfail(reason="uri format not implemented")
class TestUri:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"uri"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"http://foo.bar/?baz=qux#quux"',  # a valid URL with anchor tag
            '"http://foo.com/blah_(wikipedia)_blah#cite-1"',  # a valid URL with anchor tag and parentheses
            '"http://foo.bar/?q=Test%20URL-encoded%20stuff"',  # a valid URL with URL-encoded stuff
            '"http://xn--nw2a.xn--j6w193g/"',  # a valid puny-coded URL
            '"http://-.~_!$&\'()*+,;=:%40:80%2f::::::@example.com"',  # a valid URL with many special characters
            '"http://223.255.255.254"',  # a valid URL based on IPv4
            '"ftp://ftp.is.co.za/rfc/rfc1808.txt"',  # a valid URL with ftp scheme
            '"http://www.ietf.org/rfc/rfc2396.txt"',  # a valid URL for a simple text file
            '"ldap://[2001:db8::7]/c=GB?objectClass?one"',  # a valid URL
            '"mailto:John.Doe@example.com"',  # a valid mailto URI
            '"news:comp.infosystems.www.servers.unix"',  # a valid newsgroup URI
            '"tel:+1-816-555-1212"',  # a valid tel URI
            '"urn:oasis:names:specification:docbook:dtd:xml:4.1.2"',  # a valid URN
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"//foo.bar/?baz=qux#quux"',  # an invalid protocol-relative URI Reference
            '"/abc"',  # an invalid relative URI Reference
            '"\\\\\\\\WINDOWS\\\\fileshare"',  # an invalid URI
            '"abc"',  # an invalid URI though valid URI reference
            '"http:// shouldfail.com"',  # an invalid URI with spaces
            '":// should fail"',  # an invalid URI with spaces and missing scheme
            '"bar,baz:foo"',  # an invalid URI with comma in scheme
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


@pytest.mark.xfail(reason="uri-template format not implemented")
class TestUriTemplate:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"uri-template"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"http://example.com/dictionary/{term:1}/{term}"',  # a valid uri-template
            '"http://example.com/dictionary"',  # a valid uri-template without variables
            '"dictionary/{term:1}/{term}"',  # a valid relative uri-template
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"http://example.com/dictionary/{term:1}/{term"',  # an invalid uri-template
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


@pytest.mark.xfail(reason="iri-reference format is not yet implemented")
class TestIriReference:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"iri-reference"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"http://\\u0192\\u00f8\\u00f8.\\u00df\\u00e5r/?\\u2202\\u00e9\\u0153=\\u03c0\\u00eex#\\u03c0\\u00ee\\u00fcx"',  # a valid IRI
            '"//\\u0192\\u00f8\\u00f8.\\u00df\\u00e5r/?\\u2202\\u00e9\\u0153=\\u03c0\\u00eex#\\u03c0\\u00ee\\u00fcx"',  # a valid protocol-relative IRI Reference
            '"/\\u00e2\\u03c0\\u03c0"',  # a valid relative IRI Reference
            '"\\u00e2\\u03c0\\u03c0"',  # a valid IRI Reference
            '"#\\u0192r\\u00e4gm\\u00eant"',  # a valid IRI fragment
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"\\\\\\\\WINDOWS\\\\fil\\u00eb\\u00df\\u00e5r\\u00e9"',  # an invalid IRI Reference
            '"#\\u0192r\\u00e4g\\\\m\\u00eant"',  # an invalid IRI fragment
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


@pytest.mark.xfail(reason="iri format not implemented")
class TestIri:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"iri"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"http://\\u0192\\u00f8\\u00f8.\\u00df\\u00e5r/?\\u2202\\u00e9\\u0153=\\u03c0\\u00eex#\\u03c0\\u00ee\\u00fcx"',  # a valid IRI with anchor tag
            '"http://\\u0192\\u00f8\\u00f8.com/blah_(w\\u00eek\\u00efp\\u00e9di\\u00e5)_blah#\\u00dfit\\u00e9-1"',  # a valid IRI with anchor tag and parentheses
            '"http://\\u0192\\u00f8\\u00f8.\\u00df\\u00e5r/?q=Test%20URL-encoded%20stuff"',  # a valid IRI with URL-encoded stuff
            '"http://-.~_!$&\'()*+,;=:%40:80%2f::::::@example.com"',  # a valid IRI with many special characters
            '"http://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]"',  # a valid IRI based on IPv6
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"http://2001:0db8:85a3:0000:0000:8a2e:0370:7334"',  # an invalid IRI based on IPv6
            '"/abc"',  # an invalid relative IRI Reference
            '"\\\\\\\\WINDOWS\\\\fil\\u00eb\\u00df\\u00e5r\\u00e9"',  # an invalid IRI
            '"\\u00e2\\u03c0\\u03c0"',  # an invalid IRI though valid IRI reference
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


class TestIpv4:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"ipv4"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"192.168.0.1"',  # a valid IP address
            '"87.10.0.1"',  # value without leading zero is valid
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"127.0.0.0.1"',  # an IP address with too many components
            '"256.256.256.256"',  # an IP address with out-of-range values
            '"127.0"',  # an IP address without 4 components
            '"0x7f000001"',  # an IP address as an integer
            '"2130706433"',  # an IP address as an integer (decimal)
            '"087.10.0.1"',  # invalid leading zeroes, as they are treated as octals
            '"1\\u09e87.0.0.1"',  # invalid non-ASCII '২' (a Bengali 2)
            '"192.168.1.0/24"',  # netmask is not a part of ipv4 address
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


@pytest.mark.xfail(reason="uri-reference format not implemented")
class TestUriReference:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"uri-reference"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"http://foo.bar/?baz=qux#quux"',  # a valid URI
            '"//foo.bar/?baz=qux#quux"',  # a valid protocol-relative URI Reference
            '"/abc"',  # a valid relative URI Reference
            '"abc"',  # a valid URI Reference
            '"#fragment"',  # a valid URI fragment
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"\\\\\\\\WINDOWS\\\\fileshare"',  # an invalid URI Reference
            '"#frag\\\\ment"',  # an invalid URI fragment
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


class TestTime:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"time"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"08:30:06Z"',  # a valid time string
            '"23:59:60Z"',  # a valid time string with leap second, Zulu
            '"23:59:60+00:00"',  # valid leap second, zero time-offset
            '"01:29:60+01:30"',  # valid leap second, positive time-offset
            '"23:29:60+23:30"',  # valid leap second, large positive time-offset
            '"15:59:60-08:00"',  # valid leap second, negative time-offset
            '"00:29:60-23:30"',  # valid leap second, large negative time-offset
            '"23:20:50.52Z"',  # a valid time string with second fraction
            '"08:30:06.283185Z"',  # a valid time string with precise second fraction
            '"08:30:06+00:20"',  # a valid time string with plus offset
            '"08:30:06-08:00"',  # a valid time string with minus offset
            '"08:30:06z"',  # a valid time string with case-insensitive Z
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"008:030:006Z"',  # invalid time string with extra leading zeros
            '"8:3:6Z"',  # invalid time string with no leading zero for single digit
            '"8:0030:6Z"',  # hour, minute, second must be two digits
            pytest.param(
                '"22:59:60Z"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # invalid leap second, Zulu (wrong hour)
            pytest.param(
                '"23:58:60Z"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # invalid leap second, Zulu (wrong minute)
            pytest.param(
                '"22:59:60+00:00"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # invalid leap second, zero time-offset (wrong hour)
            pytest.param(
                '"23:58:60+00:00"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # invalid leap second, zero time-offset (wrong minute)
            pytest.param(
                '"23:59:60+01:00"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # invalid leap second, positive time-offset (wrong hour)
            pytest.param(
                '"23:59:60+00:30"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # invalid leap second, positive time-offset (wrong minute)
            pytest.param(
                '"23:59:60-01:00"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # invalid leap second, negative time-offset (wrong hour)
            pytest.param(
                '"23:59:60-00:30"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # invalid leap second, negative time-offset (wrong minute)
            '"08:30:06-8:000"',  # hour, minute in time-offset must be two digits
            '"24:00:00Z"',  # an invalid time string with invalid hour
            '"00:60:00Z"',  # an invalid time string with invalid minute
            '"00:00:61Z"',  # an invalid time string with invalid second
            pytest.param(
                '"22:59:60Z"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # an invalid time string with invalid leap second (wrong hour)
            pytest.param(
                '"23:58:60Z"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # an invalid time string with invalid leap second (wrong minute)
            '"01:02:03+24:00"',  # an invalid time string with invalid time numoffset hour
            '"01:02:03+00:60"',  # an invalid time string with invalid time numoffset minute
            '"01:02:03Z+00:30"',  # an invalid time string with invalid time with both Z and numoffset
            '"08:30:06 PST"',  # an invalid offset indicator
            '"01:01:01,1111"',  # only RFC3339 not all of ISO 8601 are valid
            '"12:00:00"',  # no time offset
            '"12:00:00.52"',  # no time offset with second fraction
            '"1\\u09e8:00:00Z"',  # invalid non-ASCII '২' (a Bengali 2)
            '"08:30:06#00:20"',  # offset not starting with plus or minus
            '"ab:cd:ef"',  # contains letters
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


class TestIpv6:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"ipv6"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"::1"',  # a valid IPv6 address
            '"::abef"',  # trailing 4 hex symbols is valid
            '"::"',  # no digits is valid
            '"::42:ff:1"',  # leading colons is valid
            '"d6::"',  # trailing colons is valid
            '"1:d6::42"',  # single set of double colons in the middle is valid
            pytest.param(
                '"1::d6:192.168.0.1"',
                marks=pytest.mark.xfail(reason="Mixed format IPv6 not implemented"),
            ),  # mixed format with the ipv4 section as decimal octets
            pytest.param(
                '"1:2::192.168.0.1"',
                marks=pytest.mark.xfail(reason="Mixed format IPv6 not implemented"),
            ),  # mixed format with double colons between the sections
            pytest.param(
                '"::ffff:192.168.0.1"',
                marks=pytest.mark.xfail(reason="Mixed format IPv6 not implemented"),
            ),  # mixed format with leading double colons (ipv4-mapped ipv6 address)
            '"1:2:3:4:5:6:7:8"',  # 8 octets
            pytest.param(
                '"1000:1000:1000:1000:1000:1000:255.255.255.255"',
                marks=pytest.mark.xfail(reason="Mixed format IPv6 not implemented"),
            ),  # a long valid ipv6
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"12345::"',  # an IPv6 address with out-of-range values
            '"::abcef"',  # trailing 5 hex symbols is invalid
            '"1:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1"',  # an IPv6 address with too many components
            '"::laptop"',  # an IPv6 address containing illegal characters
            '":2:3:4:5:6:7:8"',  # missing leading octet is invalid
            '"1:2:3:4:5:6:7:"',  # missing trailing octet is invalid
            '":2:3:4::8"',  # missing leading octet with omitted octets later
            '"1::d6::42"',  # two sets of double colons is invalid
            '"1::2:192.168.256.1"',  # mixed format with ipv4 section with octet out of range
            '"1::2:192.168.ff.1"',  # mixed format with ipv4 section with a hex octet
            '"1:2:3:4:5:::8"',  # triple colons is invalid
            '"1:2:3:4:5:6:7"',  # insufficient octets without double colons
            '"1"',  # no colons is invalid
            '"127.0.0.1"',  # ipv4 is not ipv6
            '"1:2:3:4:1.2.3"',  # ipv4 segment must have 4 octets
            '"  ::1"',  # leading whitespace is invalid
            '"::1  "',  # trailing whitespace is invalid
            '"fe80::/64"',  # netmask is not a part of ipv6 address
            '"fe80::a%eth1"',  # zone id is not a part of ipv6 address
            '"100:100:100:100:100:100:255.255.255.255.255"',  # a long invalid ipv6, below length limit, first
            '"100:100:100:100:100:100:100:255.255.255.255"',  # a long invalid ipv6, below length limit, second
            '"1:2:3:4:5:6:7:\\u09ea"',  # invalid non-ASCII '৪' (a Bengali 4)
            '"1:2::192.16\\u09ea.0.1"',  # invalid non-ASCII '৪' (a Bengali 4) in the IPv4 portion
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


class TestUnknown:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"unknown"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # unknown formats ignore integers
            "13.7",  # unknown formats ignore floats
            "{}",  # unknown formats ignore objects
            "[]",  # unknown formats ignore arrays
            "false",  # unknown formats ignore booleans
            "null",  # unknown formats ignore nulls
            '"string"',  # unknown formats ignore strings
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)


class TestHostname:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"hostname"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"www.example.com"',  # a valid host name
            '"xn--4gbwdl.xn--wgbh1c"',  # a valid punycoded IDN hostname
            '"abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijk.com"',  # maximum label length
            '"hostname"',  # single label
            '"host-name"',  # single label with hyphen
            '"h0stn4me"',  # single label with digits
            '"1host"',  # single label starting with digit
            '"hostnam3"',  # single label ending with digit
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"-a-host-name-that-starts-with--"',  # a host name starting with an illegal character
            '"not_a_valid_host_name"',  # a host name containing illegal characters
            '"a-vvvvvvvvvvvvvvvveeeeeeeeeeeeeeeerrrrrrrrrrrrrrrryyyyyyyyyyyyyyyy-long-host-name-component"',  # a host name with a component too long
            '"-hostname"',  # starts with hyphen
            '"hostname-"',  # ends with hyphen
            '"_hostname"',  # starts with underscore
            '"hostname_"',  # ends with underscore
            '"host_name"',  # contains underscore
            '"abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijkl.com"',  # exceeds maximum label length
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


class TestUuid:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"uuid"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"2EB8AA08-AA98-11EA-B4AA-73B441D16380"',  # all upper-case
            '"2eb8aa08-aa98-11ea-b4aa-73b441d16380"',  # all lower-case
            '"2eb8aa08-AA98-11ea-B4Aa-73B441D16380"',  # mixed case
            '"00000000-0000-0000-0000-000000000000"',  # all zeroes is valid
            '"98d80576-482e-427f-8434-7f86890ab222"',  # valid version 4
            '"99c17cbb-656f-564a-940f-1a4568f03487"',  # valid version 5
            '"99c17cbb-656f-664a-940f-1a4568f03487"',  # hypothetical version 6
            '"99c17cbb-656f-f64a-940f-1a4568f03487"',  # hypothetical version 15
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"2eb8aa08-aa98-11ea-b4aa-73b441d1638"',  # wrong length
            '"2eb8aa08-aa98-11ea-73b441d16380"',  # missing section
            '"2eb8aa08-aa98-11ea-b4ga-73b441d16380"',  # bad characters (not hex)
            '"2eb8aa08aa9811eab4aa73b441d16380"',  # no dashes
            '"2eb8aa08aa98-11ea-b4aa73b441d16380"',  # too few dashes
            '"2eb8-aa08-aa98-11ea-b4aa73b44-1d16380"',  # too many dashes
            '"2eb8aa08aa9811eab4aa73b441d16380----"',  # dashes in the wrong spot
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


class TestEmail:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"email"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"joe.bloggs@example.com"',  # a valid e-mail address
            '"te~st@example.com"',  # tilde in local part is valid
            '"~test@example.com"',  # tilde before local part is valid
            '"test~@example.com"',  # tilde after local part is valid
            pytest.param(
                '"\\"joe bloggs\\"@example.com"',
                marks=pytest.mark.xfail(reason="Quoted strings not yet implemented in local part"),
            ),  # a quoted string with a space in the local part is valid
            pytest.param(
                '"\\"joe..bloggs\\"@example.com"',
                marks=pytest.mark.xfail(reason="Quoted strings not yet implemented in local part"),
            ),  # a quoted string with a double dot in the local part is valid
            pytest.param(
                '"\\"joe@bloggs\\"@example.com"',
                marks=pytest.mark.xfail(reason="Quoted strings not yet implemented in local part"),
            ),  # a quoted string with a @ in the local part is valid
            '"joe.bloggs@[127.0.0.1]"',  # an IPv4-address-literal after the @ is valid
            pytest.param(
                '"joe.bloggs@[IPv6:::1]"', marks=pytest.mark.xfail(reason="IPv6 is hard")
            ),  # an IPv6-address-literal after the @ is valid
            '"te.s.t@example.com"',  # two separated dots inside local part are valid
            '"riedgar+guidance@example.com"',  # plus sign in local part is valid
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"2962"',  # an invalid e-mail address
            '".test@example.com"',  # dot before local part is not valid
            '"test.@example.com"',  # dot after local part is not valid
            '"te..st@example.com"',  # two subsequent dots inside local part are not valid
            '"joe.bloggs@invalid=domain.com"',  # an invalid domain
            '"joe.bloggs@[127.0.0.300]"',  # an invalid IPv4-address-literal
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


class TestDuration:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"duration"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"P4DT12H30M5S"',  # a valid duration string
            '"P4Y"',  # four years duration
            '"PT0S"',  # zero time, in seconds
            '"P0D"',  # zero time, in days
            '"P1M"',  # one month duration
            '"PT1M"',  # one minute duration
            '"PT36H"',  # one and a half days, in hours
            '"P1DT12H"',  # one and a half days, in days and hours
            '"P2W"',  # two weeks
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"PT1D"',  # an invalid duration string
            '"P"',  # no elements present
            '"P1YT"',  # no time elements present
            '"PT"',  # no date or time elements present
            '"P2D1Y"',  # elements out of order
            '"P1D2H"',  # missing time separator
            '"P2S"',  # time element in the date position
            '"P1Y2W"',  # weeks cannot be combined with other units
            '"P\\u09e8Y"',  # invalid non-ASCII '২' (a Bengali 2)
            '"P1"',  # element without unit
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


@pytest.mark.xfail(reason="relative-json-pointer format not implemented")
class TestRelativeJsonPointer:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"relative-json-pointer"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"1"',  # a valid upwards RJP
            '"0/foo/bar"',  # a valid downwards RJP
            '"2/0/baz/1/zip"',  # a valid up and then down RJP, with array index
            '"0#"',  # a valid RJP taking the member or index name
            '"120/foo/bar"',  # multi-digit integer prefix
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"/foo/bar"',  # an invalid RJP that is a valid JSON Pointer
            '"-1/foo/bar"',  # negative prefix
            '"+1/foo/bar"',  # explicit positive prefix
            '"0##"',  # ## is not a valid json-pointer
            '"01/a"',  # zero cannot be followed by other digits, plus json-pointer
            '"01#"',  # zero cannot be followed by other digits, plus octothorpe
            '""',  # empty string
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


class TestDateTime:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"date-time"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"1963-06-19T08:30:06.283185Z"',  # a valid date-time string
            '"1963-06-19T08:30:06Z"',  # a valid date-time string without second fraction
            '"1937-01-01T12:00:27.87+00:20"',  # a valid date-time string with plus offset
            '"1990-12-31T15:59:50.123-08:00"',  # a valid date-time string with minus offset
            '"1998-12-31T23:59:60Z"',  # a valid date-time with a leap second, UTC
            '"1998-12-31T15:59:60.123-08:00"',  # a valid date-time with a leap second, with minus offset
            '"1963-06-19t08:30:06.283185z"',  # case-insensitive T and Z
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"1998-12-31T23:59:61Z"',  # an invalid date-time past leap second, UTC
            pytest.param(
                '"1998-12-31T23:58:60Z"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # an invalid date-time with leap second on a wrong minute, UTC
            pytest.param(
                '"1998-12-31T22:59:60Z"', marks=pytest.mark.xfail(reason="leap seconds are hard")
            ),  # an invalid date-time with leap second on a wrong hour, UTC
            pytest.param(
                '"1990-02-31T15:59:59.123-08:00"',
                marks=pytest.mark.xfail(reason="valid days not yet tied to month"),
            ),  # an invalid day in date-time string
            '"1990-12-31T15:59:59-24:00"',  # an invalid offset in date-time string
            '"1963-06-19T08:30:06.28123+01:00Z"',  # an invalid closing Z after time-zone offset
            '"06/19/1963 08:30:06 PST"',  # an invalid date-time string
            '"2013-350T01:01:01"',  # only RFC3339 not all of ISO 8601 are valid
            '"1963-6-19T08:30:06.283185Z"',  # invalid non-padded month dates
            '"1963-06-1T08:30:06.283185Z"',  # invalid non-padded day dates
            '"1963-06-1\\u09eaT00:00:00Z"',  # invalid non-ASCII '৪' (a Bengali 4) in date portion
            '"1963-06-11T0\\u09ea:00:00Z"',  # invalid non-ASCII '৪' (a Bengali 4) in time portion
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


@pytest.mark.xfail(reason="regex format not implemented")
class TestRegex:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"regex"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"([abc])+\\\\s+$"',  # a valid regular expression
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"^(abc]"',  # a regular expression with unclosed parens is invalid
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)


@pytest.mark.xfail(reason="idn-email format not implemented")
class TestIdnEmail:
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema","format":"idn-email"}'

    @pytest.mark.parametrize(
        "target_str",
        [
            "12",  # all string formats ignore integers
            "13.7",  # all string formats ignore floats
            "{}",  # all string formats ignore objects
            "[]",  # all string formats ignore arrays
            "false",  # all string formats ignore booleans
            "null",  # all string formats ignore nulls
            '"\\uc2e4\\ub840@\\uc2e4\\ub840.\\ud14c\\uc2a4\\ud2b8"',  # a valid idn e-mail (example@example.test in Hangul)
            '"joe.bloggs@example.com"',  # a valid e-mail address
        ],
    )
    def test_good(self, target_str):
        schema_obj = json.loads(self.schema)
        target_obj = json.loads(target_str)
        generate_and_check(target_obj, schema_obj)

    @pytest.mark.parametrize(
        "bad_str",
        [
            '"2962"',  # an invalid idn e-mail address
            '"2962"',  # an invalid e-mail address
        ],
    )
    def test_bad(self, bad_str):
        schema_obj = json.loads(self.schema)
        check_match_failure(bad_string=bad_str, schema_obj=schema_obj)
