import requests
import os
import base64

from ._model import Engine, EngineCallResponse

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
            "grammar": base64.b64encode(grammar.serialize()).decode('utf-8')
        }

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Send the request to the server
        response = requests.post(self.server_url + "/extend", json=data, headers=headers, stream=True, verify=self.verify_crt)

        # Check for valid response
        if response.status_code != 200:
            response.raise_for_status()

        # Process and yield the response data
        for chunk in response.iter_content(chunk_size=None):  # chunk_size=None means it'll stream the content
            response_data = EngineCallResponse.deserialize(chunk)
            yield response_data