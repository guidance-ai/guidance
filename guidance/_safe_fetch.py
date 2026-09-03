"""Safe HTTP(S) fetch helpers that avoid SSRF via redirects and DNS rebinding."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import urllib.parse
from collections.abc import Sequence
from typing import Final

from ._uri_validation import URIValidationError, _check_ip_address, validate_uri

_MAX_REDIRECTS: Final[int] = 5


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _is_ip_literal(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _resolve_validated_ip(hostname: str, port: int, original_url: str, *, allow_private: bool) -> str:
    """Resolve hostname once; optionally reject private addresses."""
    if _is_ip_literal(hostname):
        addr = ipaddress.ip_address(hostname)
        if not allow_private:
            _check_ip_address(addr, original_url)
        return hostname

    try:
        addrinfos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise URIValidationError(f"Failed to resolve hostname '{hostname}': {e}") from e

    if not addrinfos:
        raise URIValidationError(f"Hostname '{hostname}' resolved to no addresses: {original_url}")

    ips: list[str] = []
    for addrinfo in addrinfos:
        ip_str = addrinfo[4][0]
        if not allow_private:
            _check_ip_address(ipaddress.ip_address(ip_str), original_url)
        ips.append(ip_str)

    return ips[0]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that dials a pre-resolved IP while preserving SNI/Host."""

    def __init__(
        self,
        hostname: str,
        pinned_ip: str,
        port: int,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(hostname, port, timeout=timeout, context=context)
        self._pinned_ip = pinned_ip
        self._ssl_context = context
        self._server_hostname = hostname

    def connect(self) -> None:
        sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        self.sock = self._ssl_context.wrap_socket(sock, server_hostname=self._server_hostname)


def fetch_url_bytes(
    url: str,
    *,
    allowed_schemes: Sequence[str],
    allow_private: bool,
    allow_local: bool,
    max_bytes: int,
    timeout: float,
) -> bytes:
    """Fetch URL bytes with redirect re-validation and DNS pinning.

    - Validates the initial URL and every redirect target via ``validate_uri``.
    - Resolves DNS once per hop and connects to that IP (with TLS SNI / Host of
      the original hostname) so a later DNS rebinding cannot point at a private IP.
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        validate_uri(
            current,
            allowed_schemes=allowed_schemes,
            allow_private=allow_private,
            allow_local=allow_local,
        )
        parsed = urllib.parse.urlparse(current)
        scheme = parsed.scheme.lower()
        if scheme == "file":
            raise URIValidationError(f"file:// must be handled by the caller, not fetch_url_bytes: {current}")

        hostname = parsed.hostname
        if hostname is None:
            raise URIValidationError(f"URI has no hostname: {current}")

        port = parsed.port or _default_port(scheme)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        pinned_ip = _resolve_validated_ip(hostname, port, current, allow_private=allow_private)

        status, headers, body = _http_get_pinned(
            scheme=scheme,
            hostname=hostname,
            pinned_ip=pinned_ip,
            port=port,
            path=path,
            timeout=timeout,
            max_bytes=max_bytes,
        )

        if 300 <= status < 400:
            location = headers.get("location")
            if not location:
                raise URIValidationError(f"Redirect response missing Location header: {current}")
            current = urllib.parse.urljoin(current, location)
            continue

        if status >= 400:
            raise URIValidationError(f"HTTP {status} fetching {current}")

        return body

    raise URIValidationError(f"Too many redirects (>{_MAX_REDIRECTS}) fetching {url}")


def _http_get_pinned(
    *,
    scheme: str,
    hostname: str,
    pinned_ip: str,
    port: int,
    path: str,
    timeout: float,
    max_bytes: int,
) -> tuple[int, dict[str, str], bytes]:
    headers = {"Host": hostname, "User-Agent": "guidance"}

    if scheme == "https":
        context = ssl.create_default_context()
        conn: http.client.HTTPConnection = _PinnedHTTPSConnection(
            hostname, pinned_ip, port, timeout, context
        )
    elif scheme == "http":
        conn = http.client.HTTPConnection(pinned_ip, port, timeout=timeout)
    else:
        raise URIValidationError(f"Unsupported scheme for fetch: {scheme}")

    try:
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        body = response.read(max_bytes + 1)
        hdrs = {k.lower(): v for k, v in response.getheaders()}
        return response.status, hdrs, body
    finally:
        conn.close()
