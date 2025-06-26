import importlib.resources
import pathlib
import typing
import base64


from .._guidance import guidance
from .._utils import bytes_from

# from ..trace._trace import VideoInput
from ..trace._trace import VideoOutput


@guidance
def video(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    bytes_data = bytes_from(src, allow_local=allow_local)
    base64_bytes = base64.b64encode(bytes_data)
    lm += VideoOutput(value=base64_bytes, is_input=True)
    # lm += VideoInput(value=base64_string)
    return lm


@guidance
def gen_video(lm):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    with importlib.resources.files("guidance").joinpath("resources/sample_video.png").open("rb") as f:
        bytes_data = f.read()
    base64_bytes = base64.b64encode(bytes_data)
    lm += VideoOutput(value=base64_bytes, is_input=False)
    return lm
