import requests
from pydantic import BaseModel
from typing import Generator
import os

from ._model import Engine, EngineCallResponse

# The response model for the server's responses
class EngineCallResponse(BaseModel):
    new_bytes: bytes
    is_generated: bool
    new_bytes_prob: float
    capture_groups: dict
    capture_group_log_probs: dict
    new_token_count: int


class RemoteEngine(Engine):
    '''This connects to a remote guidance server and runs all computation using the remote engine.'''
    def __init__(self, server_url, api_key, verify=None):
        self.server_url = server_url
        self.api_key = api_key
        if verify is None:
            verify = os.getenv("GUIDANCE_SSL_CERTFILE", None)
        self.verify_crt = verify

    def __call__(self, parser, grammar, ensure_bos_token=True):
        # Prepare the request data
        data = {
            "parser": parser,
            "grammar": grammar.serialize()
        }

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Send the request to the server
        response = requests.post(self.server_url + "/extend", json=data, headers=headers, stream=True, verify=self.verify_crt)

        # Check for valid response
        if response.status_code != 200:
            raise Exception(f"Server returned an error: {response.status_code} - {response.text}")

        # Process and yield the response data
        for chunk in response.iter_content(chunk_size=None):  # chunk_size=None means it'll stream the content
            response_data = EngineCallResponse.parse_raw(chunk)
            yield response_data