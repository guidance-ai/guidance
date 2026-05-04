import base64
import importlib.resources
import pathlib
from collections.abc import Sequence

from .._guidance import guidance
from .._uri_validation import DEFAULT_ALLOWED_SCHEMES
from .._utils import _DEFAULT_MAX_BYTES, _DEFAULT_TIMEOUT, bytes_from

# from ..trace._trace import VideoInput
from ..trace._trace import VideoOutput


@guidance
def video(
    lm,
    src: str | pathlib.Path | bytes,
    allow_local: bool = True,
    allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES,
    allow_private: bool = False,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: float = _DEFAULT_TIMEOUT,
):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    bytes_data = bytes_from(
        src,
        allow_local=allow_local,
        allowed_schemes=allowed_schemes,
        allow_private=allow_private,
        max_bytes=max_bytes,
        timeout=timeout,
    )
    base64_bytes = base64.b64encode(bytes_data)
    lm += VideoOutput(value=base64_bytes, is_input=True)
    # lm += VideoInput(value=base64_string)
    return lm


@guidance
def gen_video(lm):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    with importlib.resources.files("guidance").joinpath("resources/sample_video.png").open("rb") as f:
        bytes_data = f.read()
    base64_bytes = base64.b64encode(bytes_data)
    lm += VideoOutput(value=base64_bytes, is_input=False)
    return lm
