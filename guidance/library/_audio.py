import http
import pathlib
import re
import typing
import urllib

from .._guidance import guidance


@guidance
def audio(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    # TODO(nopdive): Stub for testing -- needs implementation.
    # NOTE(nopdive): src parsing is the same per modal, consider refactoring.

    # load the audio bytes
    # ...from a url
    if isinstance(src, str) and re.match(r"[^:/]+://", src):
        with urllib.request.urlopen(src) as response:
            response = typing.cast(http.client.HTTPResponse, response)
            bytes_data = response.read()

    # ...from a local path
    elif allow_local and (isinstance(src, str) or isinstance(src, pathlib.Path)):
        with open(src, "rb") as f:
            bytes_data = f.read()

    # ...from audio file bytes
    elif isinstance(src, bytes):
        bytes_data = src

    else:
        raise Exception(f"Unable to load audio bytes from {src}!")

    # Add audio to LM.
    # lm += ...
    return lm