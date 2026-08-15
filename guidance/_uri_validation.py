"""URI validation utilities to prevent SSRF and unauthorized resource access."""

import ipaddress
import socket
import urllib.parse
from collections.abc import Sequence

# Default: only HTTPS is permitted
DEFAULT_ALLOWED_SCHEMES: tuple[str, ...] = ("https",)


# Cloud metadata / credential endpoints. Most sit inside the ranges `_check_ip_address`
# already rejects, but `168.63.129.16` (Azure) and `100.100.100.200` (Alibaba) do not: the
# first is publicly routable and the second is in `100.64.0.0/10`, which Python does not
# report as private. They are listed explicitly so a range check is not the only thing
# standing between a URL and a credential endpoint.
_CLOUD_METADATA_ADDRESSES: frozenset[ipaddress.IPv4Address | ipaddress.IPv6Address] = frozenset(
    ipaddress.ip_address(ip)
    for ip in (
        "169.254.169.254",  # AWS IMDS, GCP, Azure, OCI, DigitalOcean, Hetzner, IBM, OpenStack
        "169.254.170.2",  # AWS ECS task IAM role credentials
        "169.254.170.23",  # AWS EKS Pod Identity Agent
        "168.63.129.16",  # Azure WireServer / platform channel (publicly routable)
        "100.100.100.200",  # Alibaba Cloud
        "192.0.0.192",  # Oracle Cloud (Classic)
        "169.254.42.42",  # Scaleway
        "fd00:ec2::254",  # AWS IMDS over IPv6
        "fd00:ec2::23",  # AWS EKS Pod Identity Agent over IPv6
    )
)


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

    # For network URIs, validate the host. Cloud metadata endpoints are checked even when
    # `allow_private` is set: reaching a host on your own network and reaching the instance's
    # credential endpoint are different requests, and only the first is what the flag is for.
    hostname = parsed.hostname
    if hostname is None:
        raise URIValidationError(f"URI has no hostname: {url}")
    _validate_hostname_not_private(hostname, url, allow_private=allow_private)


def _validate_hostname_not_private(hostname: str, original_url: str, allow_private: bool = False) -> None:
    """Resolve hostname and verify it doesn't point to a metadata/private/loopback/link-local address."""
    # First, check if the hostname is already an IP literal
    try:
        addr = ipaddress.ip_address(hostname)
        _check_ip_address(addr, original_url, allow_private=allow_private)
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
        _check_ip_address(addr, original_url, allow_private=allow_private)


def _check_ip_address(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address, original_url: str, allow_private: bool = False
) -> None:
    """Raise if the IP address is a cloud metadata endpoint, private, loopback, or link-local."""
    # `::ffff:168.63.129.16` and `168.63.129.16` are the same destination, so unwrap before the
    # membership test. This matters most for the two addresses the list above exists for: both are
    # publicly routable, so the range checks below never see them and equality against the list is
    # the only thing in the way. The link-local entries survive the wrapper on their own because
    # Python reports an IPv4-mapped address as link-local, which is what hid this.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped

    if addr in _CLOUD_METADATA_ADDRESSES:
        raise URIValidationError(f"URI resolves to a cloud metadata endpoint ({addr}): {original_url}")
    if allow_private:
        return
    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
        raise URIValidationError(
            f"URI resolves to a private/loopback/link-local/reserved address ({addr}): {original_url}"
        )
