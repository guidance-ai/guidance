import base64
import importlib.resources
import pathlib
import re
import typing

from .._ast import ImageBlob, ImageUrl
from .._guidance import guidance
from .._utils import bytes_from
from ..trace._trace import ImageOutput


@guidance
def image(lm, src: str | pathlib.Path | bytes, allow_local: bool = True):
    if isinstance(src, str) and re.match(r"^(?!file://)[^:/]+://", src):
        lm += ImageUrl(url=src)
    else:
        bytes_data = bytes_from(src, allow_local=allow_local)
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
