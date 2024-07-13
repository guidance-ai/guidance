import http
import pathlib
import re
import typing
import urllib

from .._guidance import guidance


@guidance
def image(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):

    # load the image bytes
    # ...from a url
    if isinstance(src, str) and re.match(r"[^:/]+://", src):
        with urllib.request.urlopen(src) as response:
            response = typing.cast(http.client.HTTPResponse, response)
            bytes_data = response.read()

    # ...from a local path
    elif allow_local and (isinstance(src, str) or isinstance(src, pathlib.Path)):
        with open(src, "rb") as f:
            bytes_data = f.read()

    # ...from image file bytes
    elif isinstance(src, bytes):
        bytes_data = src

    else:
        raise Exception(f"Unable to load image bytes from {src}!")

    bytes_id = str(id(bytes_data))

    # set the image bytes
    lm = lm.set(bytes_id, bytes_data)
    lm += f"<|_image:{bytes_id}|>"
    return lm
