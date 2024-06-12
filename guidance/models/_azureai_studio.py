import hashlib
import pathlib
import urllib.parse

import diskcache as dc
import platformdirs
import requests

from ._model import Chat
from ._grammarless import GrammarlessEngine, Grammarless


try:
    import openai

    is_openai = True
except ModuleNotFoundError:
    is_openai = False


class AzureAIStudioChatEngine(GrammarlessEngine):
    def __init__(
        self,
        *,
        tokenizer,
        max_streaming_tokens: int,
        timeout: float,
        compute_log_probs: bool,
        azureai_studio_endpoint: str,
        azureai_model_deployment: str,
        azureai_studio_key: str,
        clear_cache: bool,
    ):
        endpoint_parts = urllib.parse.urlparse(azureai_studio_endpoint)
        if endpoint_parts.path == "/score":
            self._is_openai_compatible = False
            self._endpoint = azureai_studio_endpoint
        else:
            if not is_openai:
                raise ValueError(
                    "Detected OpenAI compatible model; please install openai package"
                )
            self._is_openai_compatible = True
            self._endpoint = f"{endpoint_parts.scheme}://{endpoint_parts.hostname}"
        self._deployment = azureai_model_deployment
        self._api_key = azureai_studio_key

        # There is a cache... better make sure it's specific
        # to the endpoint and deployment
        deployment_id = self._hash_prompt(self._endpoint + self._deployment)

        path = (
            pathlib.Path(platformdirs.user_cache_dir("guidance"))
            / f"azureaistudio.tokens.{deployment_id}"
        )
        self.cache = dc.Cache(path)
        if clear_cache:
            self.cache.clear()

        super().__init__(tokenizer, max_streaming_tokens, timeout, compute_log_probs)

    def _hash_prompt(self, prompt):
        # Copied from OpenAIChatEngine
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _generator(self, prompt: bytes, temperature: float):
        # Initial parts of this straight up copied from OpenAIChatEngine

        # The next loop (or one like it) appears in several places,
        # and quite possibly belongs in a library function or superclass
        # That said, I'm not _completely sure that there aren't subtle
        # differences between the various versions

        assert isinstance(prompt, bytes)

        # find the role tags
        pos = 0
        input_token_count = 0
        role_end = b"<|im_end|>"
        messages = []
        found = True
        while found:

            # find the role text blocks
            found = False
            for role_name, start_bytes in (
                ("system", b"<|im_start|>system\n"),
                ("user", b"<|im_start|>user\n"),
                ("assistant", b"<|im_start|>assistant\n"),
            ):
                if prompt[pos:].startswith(start_bytes):
                    pos += len(start_bytes)
                    end_pos = prompt[pos:].find(role_end)
                    if end_pos < 0:
                        assert (
                            role_name == "assistant"
                        ), "Bad chat format! Last role before gen needs to be assistant!"
                        break
                    btext = prompt[pos : pos + end_pos]
                    pos += end_pos + len(role_end)
                    message_content = btext.decode("utf8")
                    input_token_count += len(self.tokenizer.encode(btext))
                    messages.append({"role": role_name, "content": message_content})
                    found = True
                    break

        # Add nice exception if no role tags were used in the prompt.
        # TODO: Move this somewhere more general for all chat models?
        if messages == []:
            raise ValueError(
                f"The model is a Chat-based model and requires role tags in the prompt! \
            Make sure you are using guidance context managers like `with system():`, `with user():` and `with assistant():` \
            to appropriately format your guidance program for this type of model."
            )

        # Update shared data state
        self._reset_shared_data(prompt[:pos], temperature)

        # Use cache only when temperature is 0
        if temperature == 0:
            cache_key = self._hash_prompt(prompt)

            # Check if the result is already in the cache
            if cache_key in self.cache:
                for chunk in self.cache[cache_key]:
                    yield chunk
                return

        # Call the actual API and extract the next chunk
        if self._is_openai_compatible:
            client = openai.OpenAI(api_key=self._api_key, base_url=self._endpoint)
            response = client.chat.completions.create(
                model=self._deployment,
                messages=messages,  # type: ignore[arg-type]
                # max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=1.0,  # TODO: this should be controllable like temp (from the grammar)
                temperature=temperature,
                # stream=True,
            )

            result = response.choices[0]
            chunk = result.message.content
            encoded_chunk = chunk.encode("utf8")  # type: ignore[union-attr]

            # Non-streaming OpenAI call, so we can just get the metrics directly
            if response.usage is not None:
                self.metrics.engine_input_tokens += response.usage.prompt_tokens
                self.metrics.engine_output_tokens += response.usage.completion_tokens
        else:
            parameters = dict(temperature=temperature)
            payload = dict(
                input_data=dict(input_string=messages, parameters=parameters)
            )

            headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self._api_key),
                "azureml-model-deployment": self._deployment,
            }
            response_score = requests.post(
                self._endpoint,
                json=payload,
                headers=headers,
            )

            result_score = response_score.json()

            chunk = result_score["output"]
            encoded_chunk = chunk.encode("utf8")
            self.metrics.engine_input_tokens += input_token_count
            self.metrics.engine_output_tokens += len(
                self.tokenizer.encode(encoded_chunk)
            )

        # Now back to OpenAIChatEngine, with slight modifications since
        # this isn't a streaming API
        if temperature == 0:
            cached_results = []

        yield encoded_chunk

        if temperature == 0:
            cached_results.append(encoded_chunk)

        # Cache the results after the generator is exhausted
        if temperature == 0:
            self.cache[cache_key] = cached_results


class AzureAIStudioChat(Grammarless, Chat):
    def __init__(
        self,
        azureai_studio_endpoint: str,
        azureai_studio_deployment: str,
        azureai_studio_key: str,
        tokenizer=None,
        echo: bool = True,
        max_streaming_tokens: int = 1000,
        timeout: float = 0.5,
        compute_log_probs: bool = False,
        clear_cache: bool = False,
    ):
        """Create a model object for interacting with Azure AI Studio chat endpoints.

        The required information about the deployed endpoint can
        be obtained from Azure AI Studio.

        A `diskcache`-based caching system is used to speed up
        repeated calls when the temperature is specified to be
        zero.

        Parameters
        ----------
        azureai_studio_endpoint : str
            The HTTPS endpoint deployed by Azure AI Studio
        azureai_studio_deployment : str
            The specific model deployed to the endpoint
        azureai_studio_key : str
            The key required for access to the API
        clear_cache : bool
            Whether to empty the internal cache
        """
        super().__init__(
            AzureAIStudioChatEngine(
                azureai_studio_endpoint=azureai_studio_endpoint,
                azureai_model_deployment=azureai_studio_deployment,
                azureai_studio_key=azureai_studio_key,
                tokenizer=tokenizer,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
                clear_cache=clear_cache,
            ),
            echo=echo,
        )
