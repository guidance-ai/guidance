import pathlib
import typing
import base64

from .._guidance import guidance
from .._utils import bytes_from
from .._ast import AudioBlob, GenAudio


@guidance
def audio(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    bytes_data = bytes_from(src, allow_local=allow_local)
    base64_string = base64.b64encode(bytes_data).decode('utf-8')
    lm += AudioBlob(data=base64_string)
    return lm


@guidance
def gen_audio(lm, **kwargs):
    return lm + GenAudio(kwargs=kwargs)
