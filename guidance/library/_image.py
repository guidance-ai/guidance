import base64
import importlib.resources
import pathlib
import re
from collections.abc import Sequence

from .._ast import ImageBlob, ImageUrl
from .._guidance import guidance
from .._uri_validation import DEFAULT_ALLOWED_SCHEMES, validate_uri
from .._utils import _DEFAULT_MAX_BYTES, _DEFAULT_TIMEOUT, bytes_from
from ..trace._trace import ImageOutput


@guidance
def image(
    lm,
    src: str | pathlib.Path | bytes,
    allow_local: bool = True,
    allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES,
    allow_private: bool = False,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: float = _DEFAULT_TIMEOUT,
):
    if isinstance(src, str) and re.match(r"^(?!file://)[^:/]+://", src):
        validate_uri(src, allowed_schemes=allowed_schemes, allow_private=allow_private, allow_local=allow_local)
        lm += ImageUrl(url=src)
    else:
        bytes_data = bytes_from(
            src,
            allow_local=allow_local,
            allowed_schemes=allowed_schemes,
            allow_private=allow_private,
            max_bytes=max_bytes,
            timeout=timeout,
        )
        base64_bytes = base64.b64encode(bytes_data)
        lm += ImageBlob(data=base64_bytes)
    return lm


@guidance
def gen_image(lm):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    with importlib.resources.files("guidance").joinpath("resources/sample_image.png").open("rb") as f:
        bytes_data = f.read()
    base64_bytes = base64.b64encode(bytes_data)
    lm += ImageOutput(value=base64_bytes, is_input=False)
    return lm
