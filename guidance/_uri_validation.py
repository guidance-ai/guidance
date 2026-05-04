"""URI validation utilities to prevent SSRF and unauthorized resource access."""

import ipaddress
import socket
import urllib.parse
from collections.abc import Sequence

# Default: only HTTPS is permitted
DEFAULT_ALLOWED_SCHEMES: tuple[str, ...] = ("https",)


class URIValidationError(ValueError):
    """Raised when a URI fails security validation."""

    pass


def validate_uri(
    url: str,
    *,
    allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES,
    allow_private: bool = False,
    allow_local: bool = True,
) -> None:
    """Validate a URI against security policies.

    Parameters
    ----------
    url : str
        The URI to validate.
    allowed_schemes : Sequence[str]
        Permitted URI schemes (without '://'). Default is ("https",).
    allow_private : bool
        If False (default), block resolution to private/loopback/link-local IPs.
    allow_local : bool
        If False, block file:// URIs.

    Raises
    ------
    URIValidationError
        If the URI violates any security policy.
    """
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()

    # Block file:// when allow_local is False
    if scheme == "file" and not allow_local:
        raise URIValidationError(f"file:// URIs are not allowed when allow_local=False: {url}")

    # Check scheme allowlist (file:// is handled separately via allow_local)
    if scheme == "file":
        # file:// is governed by allow_local, not the scheme allowlist
        return

    if scheme not in [s.lower() for s in allowed_schemes]:
        raise URIValidationError(f"URI scheme '{scheme}' is not in allowed schemes {list(allowed_schemes)}: {url}")

    # For network URIs, validate the host against private IP ranges
    if not allow_private:
        hostname = parsed.hostname
        if hostname is None:
            raise URIValidationError(f"URI has no hostname: {url}")
        _validate_hostname_not_private(hostname, url)


def _validate_hostname_not_private(hostname: str, original_url: str) -> None:
    """Resolve hostname and verify it doesn't point to a private/loopback/link-local address."""
    # First, check if the hostname is already an IP literal
    try:
        addr = ipaddress.ip_address(hostname)
        _check_ip_address(addr, original_url)
        return
    except ValueError:
        pass  # Not an IP literal, proceed with DNS resolution

    # Resolve the hostname to IP addresses
    try:
        addrinfos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise URIValidationError(f"Failed to resolve hostname '{hostname}': {e}") from e

    if not addrinfos:
        raise URIValidationError(f"Hostname '{hostname}' resolved to no addresses: {original_url}")

    # Check ALL resolved addresses (prevent DNS rebinding via multiple A records)
    for addrinfo in addrinfos:
        ip_str = addrinfo[4][0]
        addr = ipaddress.ip_address(ip_str)
        _check_ip_address(addr, original_url)


def _check_ip_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address, original_url: str) -> None:
    """Raise if the IP address is private, loopback, or link-local."""
    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
        raise URIValidationError(
            f"URI resolves to a private/loopback/link-local/reserved address ({addr}): {original_url}"
        )
