import base64
import pathlib

from .._ast import AudioBlob, GenAudio
from .._guidance import guidance
from .._utils import bytes_from


@guidance
def audio(lm, src: str | pathlib.Path | bytes, allow_local: bool = True):
    bytes_data = bytes_from(src, allow_local=allow_local)
    base64_bytes = base64.b64encode(bytes_data)
    lm += AudioBlob(data=base64_bytes)
    return lm


@guidance
def gen_audio(lm, **kwargs):
    return lm + GenAudio(kwargs=kwargs)
