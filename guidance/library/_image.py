import pathlib
import typing

from .._guidance import guidance
from .._utils import bytes_from


@guidance
def image(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    bytes_data = bytes_from(src, allow_local=allow_local)
    bytes_id = str(id(bytes_data))

    # set the image bytes
    lm = lm.set(bytes_id, bytes_data)
    lm += f"<|_image:{bytes_id}|>"
    return lm
