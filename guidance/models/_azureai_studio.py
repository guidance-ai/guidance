import json
import urllib.request

from ._model import Chat
from ._grammarless import GrammarlessEngine, Grammarless


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
    ):
        self._endpoint = azureai_studio_endpoint
        self._deployment = azureai_model_deployment
        self._api_key = azureai_studio_key

        super().__init__(tokenizer, max_streaming_tokens, timeout, compute_log_probs)

    def _generator(self, prompt, temperature: float):
        # Initial parts of this straight up copied from OpenAIChatEngine

        # find the role tags
        pos = 0
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
                    messages.append(
                        {"role": role_name, "content": btext.decode("utf8")}
                    )
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

        # Now switch to the example code from AzureAI Studio

        # Prepare for the API call (this might be model specific....)
        parameters = dict(temperature=temperature)
        payload = dict(input_data=dict(input_string=messages, parameters=parameters))

        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self._api_key),
            "azureml-model-deployment": self._deployment,
        }

        body = str.encode(json.dumps(payload))

        req = urllib.request.Request(self._endpoint, body, headers)

        response = urllib.request.urlopen(req)
        result = json.loads(response.read())

        # Now back to OpenAIChatEngine
        if temperature == 0:
            cached_results = []

        yield result["output"]

        if temperature == 0:
            cached_results.append(result["output"])

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
    ):
        super().__init__(
            AzureAIStudioChatEngine(
                azureai_studio_endpoint=azureai_studio_endpoint,
                azureai_model_deployment=azureai_studio_deployment,
                azureai_studio_key=azureai_studio_key,
                tokenizer=tokenizer,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
            ),
            echo=echo,
        )
