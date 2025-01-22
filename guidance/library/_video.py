import http
import pathlib
import re
import typing
import urllib

from .._guidance import guidance


@guidance
def video(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    # TODO(nopdive): Stub for testing -- needs implementation.
    # NOTE(nopdive): src parsing is the same per modal, consider refactoring.

    # load the video bytes
    # ...from a url
    if isinstance(src, str) and re.match(r"[^:/]+://", src):
        with urllib.request.urlopen(src) as response:
            response = typing.cast(http.client.HTTPResponse, response)
            bytes_data = response.read()

    # ...from a local path
    elif allow_local and (isinstance(src, str) or isinstance(src, pathlib.Path)):
        with open(src, "rb") as f:
            bytes_data = f.read()

    # ...from video file bytes
    elif isinstance(src, bytes):
        bytes_data = src

    else:
        raise Exception(f"Unable to load video bytes from {src}!")

    # Add video to LM.
    # lm += ...
    return lm
