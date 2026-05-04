import base64
import pathlib
from collections.abc import Sequence

from .._ast import AudioBlob, GenAudio
from .._guidance import guidance
from .._uri_validation import DEFAULT_ALLOWED_SCHEMES
from .._utils import _DEFAULT_MAX_BYTES, _DEFAULT_TIMEOUT, bytes_from


@guidance
def audio(
    lm,
    src: str | pathlib.Path | bytes,
    allow_local: bool = True,
    allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES,
    allow_private: bool = False,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout: float = _DEFAULT_TIMEOUT,
):
    bytes_data = bytes_from(
        src,
        allow_local=allow_local,
        allowed_schemes=allowed_schemes,
        allow_private=allow_private,
        max_bytes=max_bytes,
        timeout=timeout,
    )
    base64_bytes = base64.b64encode(bytes_data)
    lm += AudioBlob(data=base64_bytes)
    return lm


@guidance
def gen_audio(lm, **kwargs):
    return lm + GenAudio(kwargs=kwargs)
