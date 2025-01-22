import pathlib
import typing

from .._guidance import guidance
from .._utils import bytes_from


@guidance
def video(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    # TODO(nopdive): Stub for testing -- needs implementation.
    bytes_data = bytes_from(src, allow_local=allow_local)

    # Add video to LM.
    # lm += ...
    return lm
