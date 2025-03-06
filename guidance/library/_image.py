import pathlib
import typing
import pkg_resources
import base64
import re

from .._guidance import guidance
from .._utils import bytes_from
from .._ast import ImageBlob, ImageUrl
from ..trace._trace import ImageOutput


@guidance
def image(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    if isinstance(src, str) and re.match(r"^(?!file://)[^:/]+://", src):
        lm += ImageUrl(url=src)
    else:
        bytes_data = bytes_from(src, allow_local=allow_local)
        base64_string = base64.b64encode(bytes_data).decode('utf-8')
        lm += ImageBlob(data=base64_string)
    return lm


@guidance
def gen_image(lm):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    with open(
            pkg_resources.resource_filename("guidance", "resources/sample_image.png"), "rb"
    ) as f:
        bytes_data = f.read()
        base64_string = base64.b64encode(bytes_data).decode('utf-8')
        lm += ImageOutput(value=base64_string, is_input=False)
    return lm
