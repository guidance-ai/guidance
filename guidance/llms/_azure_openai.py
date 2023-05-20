import os
import atexit
import json
import platformdirs
from ._openai import OpenAI


class AzureOpenAI(OpenAI):
    """ Azure OpenAI style integration.

    Warning: This class is not finalized and may change in the future.
    """

    cache = OpenAI._open_cache("_azure_openai.diskcache")

    def __init__(self, model=None, client_id=None, authority=None, caching=True, max_retries=5, max_calls_per_min=60, token=None,
                 endpoint=None, scopes=None, temperature=0.0, chat_mode="auto"):
        

        assert endpoint is not None, "An endpoint must be specified!"
        
        # build a standard OpenAI LLM object
        super().__init__(
            model=model, caching=caching, max_retries=max_retries, max_calls_per_min=max_calls_per_min,
            token=token, endpoint=endpoint, temperature=temperature, chat_mode=chat_mode
        )

        self.client_id = client_id
        self.authority = authority
        self.scopes = scopes

        from msal import PublicClientApplication, SerializableTokenCache
        self._token_cache = SerializableTokenCache()
        self._token_cache_path = os.path.join(platformdirs.user_cache_dir("guidance"), "_azure_openai.token")
        self._app = PublicClientApplication(client_id=self.client_id, authority=self.authority, token_cache=self._token_cache)
        if os.path.exists(self._token_cache_path):
            self._token_cache.deserialize(open(self._token_cache_path, 'r').read())

        self._rest_headers["X-ModelType"] = self.model_name

    @property
    def token(self):
        return self._get_token()
    @token.setter
    def token(self, value):
        pass # ignored for now

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(self.scopes, account=chosen)
    
        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=self.scopes)

            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4))

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

            # save the aquired token
            with open(self._token_cache_path, "w") as f:
                f.write(self._token_cache.serialize())

        return result["access_token"]