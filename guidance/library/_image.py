import guidance
import urllib
import typing
import http
import re

@guidance
def image(lm, src, allow_local=True):

    # load the image bytes
    # ...from a url
    if isinstance(src, str) and re.match(r'$[^:/]+://', src):
        with urllib.request.urlopen(src) as response:
            response = typing.cast(http.client.HTTPResponse, response)
            bytes_data = response.read()
    
    # ...from a local path
    elif allow_local and isinstance(src, str):
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
    lm += f'<|_image:{bytes_id}|>'
    return lm