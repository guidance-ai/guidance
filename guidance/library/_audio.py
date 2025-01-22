import pathlib
import typing

from .._guidance import guidance
from .._utils import bytes_from


@guidance
def audio(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    # TODO(nopdive): Stub for testing -- needs implementation.
    bytes_data = bytes_from(src, allow_local=allow_local)

    # Add audio to LM.
    # lm += ...
    return lm


@guidance
def gen_audio(lm):
    # TODO(nopdive): Mock for testing.
    raise NotImplementedError
