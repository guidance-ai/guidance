import urllib.parse

import requests

from ._model import Chat
from ._grammarless import GrammarlessEngine, Grammarless


try:
    import openai

    is_openai = True
except ModuleNotFoundError:
    is_openai = False


_ENDPOINT_TYPES = ["chat", "completion"]


class AzureAIStudioEngine(GrammarlessEngine):
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
        endpoint_type: str,
    ):
        if endpoint_type not in _ENDPOINT_TYPES:
            msg = f"endpoint_type {endpoint_type} not valid"
            raise ValueError(msg)
        self.endpoint_type = endpoint_type
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

        super().__init__(tokenizer, max_streaming_tokens, timeout, compute_log_probs)

    def _generator_chat(self, prompt, temperature: float):
        # Initial parts of this straight up copied from OpenAIEngine

        # The next loop (or one like it) appears in several places,
        # and quite possibly belongs in a library function or superclass
        # That said, I'm not _completely sure that there aren't subtle
        # differences between the various versions

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
                    input_token_count += len(self.tokenizer(message_content))
                    messages.append({"role": role_name, "content": message_content})
                    found = True
                    break

        # Add nice exception if no role tags were used in the prompt.
        # TODO: Move this somewhere more general for all chat models?
        if messages == []:
            raise ValueError(
                f"AzureAIStudio currently only supports chat-based models and requires role tags in the prompt! \
            Make sure you are using guidance context managers like `with system():`, `with user():` and `with assistant():` \
            to appropriately format your guidance program for this type of model."
            )

        # Update shared data state
        self._reset_shared_data(prompt[:pos], temperature)

        # Call the actual API and extract the next chunk
        encoded_chunk = ""
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

            chunk_score = result_score["output"]
            encoded_chunk = chunk_score.encode("utf8")
            self.metrics.engine_input_tokens += input_token_count
            self.metrics.engine_output_tokens += len(self.tokenizer(chunk))

        yield encoded_chunk

    def _generator_completion(self, prompt, temperature: float):
        prompt_decoded = prompt.decode("utf8")
        parameters = dict(temperature=temperature, return_full_text=True)
        payload = dict(
            input_data=dict(input_string=[prompt_decoded], parameters=parameters)
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

        chunk = result_score[0]["0"]
        encoded_chunk = chunk.encode("utf8")
        self.metrics.engine_input_tokens += len(self.tokenizer(prompt_decoded))
        self.metrics.engine_output_tokens += len(self.tokenizer(chunk))
        yield encoded_chunk

    def _generator(self, prompt, temperature: float):
        if self.endpoint_type == "chat":
            return self._generator_chat(prompt, temperature)
        else:
            return self._generator_completion(prompt, temperature)


class AzureAIStudio(Grammarless, Chat):
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
        endpoint_type: str = "chat",
    ):
        """Create a model object for interacting with Azure AI Studio endpoints.

        The required information about the deployed endpoint can
        be obtained from Azure AI Studio.

        Parameters
        ----------
        azureai_studio_endpoint : str
            The HTTPS endpoint deployed by Azure AI Studio
        azureai_studio_deployment : str
            The specific model deployed to the endpoint
        azureai_studio_key : str
            The key required for access to the API
        endpoint_type : str
            Indicates whether the endpoint is 'chat' or 'completion'
        """
        super().__init__(
            AzureAIStudioEngine(
                azureai_studio_endpoint=azureai_studio_endpoint,
                azureai_model_deployment=azureai_studio_deployment,
                azureai_studio_key=azureai_studio_key,
                tokenizer=tokenizer,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
                endpoint_type=endpoint_type,
            ),
            echo=echo,
        )
