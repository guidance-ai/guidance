import pathlib
import typing
import base64

from .._guidance import guidance
from .._utils import bytes_from
from .._ast import GenAudio
from ..trace._trace import AudioOutput


@guidance
def audio(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    bytes_data = bytes_from(src, allow_local=allow_local)
    base64_string = base64.b64encode(bytes_data).decode('utf-8')
    lm += AudioOutput(value=base64_string, is_input=True)
    return lm


@guidance
def gen_audio(lm, **kwargs):
    return lm + GenAudio(kwargs=kwargs)
