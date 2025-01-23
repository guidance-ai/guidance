import pathlib
import typing

from .._guidance import guidance
from .._utils import bytes_from
from ..trace._trace import AudioInput


@guidance
def audio(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    bytes_data = bytes_from(src, allow_local=allow_local)
    lm += AudioInput(value=bytes_data)
    return lm


@guidance
def gen_audio(lm):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    raise NotImplementedError
