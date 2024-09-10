from .._guidance import guidance
from ._ebnf import EBNF
from functools import lru_cache

@lru_cache(maxsize=1)
def rfc3986() -> EBNF:
    # https://datatracker.ietf.org/doc/html/rfc3986#appendix-A
    gbnf = """
        !start: uri

        uri: scheme ":" hier_part ["?" query] ["#" fragment]

        hier_part: "//" authority path_abempty
            | path_absolute
            | path_rootless
        //  | path_empty

        uri_reference: uri | relative_ref
        absolute_uri: scheme ":" hier_part ["?" query]
        relative_ref: relative_part ["?" query] ["#" fragment]

        relative_part: "//" authority path_abempty
            | path_absolute
            | path_noscheme
        //  | path_empty

        scheme: ALPHA (ALPHA | DIGIT | "+" | "-" | ".")*

        authority: [userinfo "@"] host [":" port]
        userinfo: (unreserved | pct_encoded | sub_delims | ":")*
        host: ip_literal | ipv4address | reg_name
        port: DIGIT*

        ip_literal: "[" (ipv6address | ipvfuture) "]"

        ipvfuture: "v" HEXDIG+ "." (unreserved | sub_delims | ":")+

        ipv6address:                    (h16 ":")~6 ls32
            |                      "::" (h16 ":")~5 ls32
            | [               h16] "::" (h16 ":")~4 ls32
            | [(h16 ":")~0..1 h16] "::" (h16 ":")~3 ls32
            | [(h16 ":")~0..2 h16] "::" (h16 ":")~2 ls32
            | [(h16 ":")~0..3 h16] "::"  h16 ":"    ls32
            | [(h16 ":")~0..4 h16] "::"             ls32
            | [(h16 ":")~0..5 h16] "::"             h16
            | [(h16 ":")~0..6 h16] "::"

        h16: HEXDIG ~ 1..4
        ls32: (h16 ":" h16) | ipv4address
        ipv4address: dec_octet "." dec_octet "." dec_octet "." dec_octet

        dec_octet: DIGIT
                | "1".."9" DIGIT
                | "1" DIGIT DIGIT
                | "2" "0".."4" DIGIT
                | "25" "0".."5"

        reg_name: (unreserved | pct_encoded | sub_delims)*

        path: path_abempty
            | path_absolute
            | path_noscheme
            | path_rootless
        //  | path_empty

        path_abempty: ("/" segment)*
        path_absolute: "/" [segment_nz ("/" segment)*]
        path_noscheme: segment_nz_nc ("/" segment)*
        path_rootless: segment_nz ("/" segment)*
        // path_empty: ""

        segment: pchar*
        segment_nz: pchar+
        segment_nz_nc: (unreserved | pct_encoded | sub_delims | "@")+

        pchar: unreserved | pct_encoded | sub_delims | ":" | "@"

        query: (pchar | "/" | "?")*

        fragment: (pchar | "/" | "?")*

        pct_encoded: "%" HEXDIG HEXDIG

        unreserved: ALPHA | DIGIT | "-" | "." | "_" | "~"
        reserved: gen_delims | sub_delims

        gen_delims: ":" | "/" | "?" | "#" | "[" | "]" | "@"
        sub_delims: "!" | "$" | "&" | "'" | "(" | ")" | "*" | "+" | "," | ";" | "="

        ALPHA: /[A-Za-z]/
        DIGIT: /[0-9]/
        HEXDIG: /[A-Fa-f0-9]/
    """
    return EBNF.from_grammar_string(gbnf)

@guidance(stateless=True, cache=True)
def uri(lm, name=None):
    return lm + rfc3986().build(name=name, start='uri')

@guidance(stateless=True, cache=True)
def uri_reference(lm, name=None):
    return lm + rfc3986().build(name=name, start='uri_reference')

@guidance(stateless=True, cache=True)
def ipv4address(lm, name=None):
    return lm + rfc3986().build(name=name, start='ipv4address')

@guidance(stateless=True, cache=True)
def ipv6address(lm, name=None):
    return lm + rfc3986().build(name=name, start='ipv6address')