import os
import tiktoken

from ._model import Chat, Instruct
from ._grammarless import GrammarlessEngine, Grammarless


class AnthropicEngine(GrammarlessEngine):
    def __init__(
        self,
        model,
        tokenizer,
        api_key,
        timeout,
        max_streaming_tokens,
        compute_log_probs,
        **kwargs,
    ):
        try:
            from anthropic import Anthropic
        except ModuleNotFoundError:
            raise Exception(
                "Please install the anthropic package version >= 0.7 using `pip install anthropic -U` in order to use guidance.models.Anthropic!"
            )

        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if api_key is None:
            raise Exception(
                "Expected an api_key argument or the ANTHROPIC_API_KEY environment variable to be set!"
            )

        self.anthropic = Anthropic(api_key=api_key, **kwargs)
        self.model_name = model

        # we pretend it tokenizes like gpt2 if tiktoken does not know about it... TODO: make this better
        if tokenizer is None:
            try:
                tokenizer = tiktoken.encoding_for_model(model)
            except:
                tokenizer = tiktoken.get_encoding("gpt2")

        super().__init__(tokenizer, max_streaming_tokens, timeout, compute_log_probs)

    def _generator(self, prompt, temperature):

        # find the role tags
        pos = 0
        role_end = b"<|im_end|>\n"
        messages = []
        found = True
        system_prompt = None # Not mandatory, but we'll store it if found
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
                    if role_name == "system":
                        system_prompt = btext.decode("utf8")
                    else:
                        messages.append(
                            {"role": role_name, "content": btext.decode("utf8")}
                        )
                    found = True
                    break

        # Add nice exception if no role tags were used in the prompt.
        # TODO: Move this somewhere more general for all chat models?
        if messages == []:
            raise ValueError(
                f"The AnthropicAI model {self.model_name} is a Chat-based model and requires role tags in the prompt! \
            Make sure you are using guidance context managers like `with system():`, `with user():` and `with assistant():` \
            to appropriately format your guidance program for this type of model."
            )

        # update our shared data state
        self._reset_shared_data(prompt, temperature)

        # API call and response handling
        try:
            # Need to do this because Anthropic API is a bit weird with the system keyword...
            model_kwargs = dict(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_streaming_tokens,
                temperature=temperature,
            )
            if system_prompt is not None:
                model_kwargs["system"] = system_prompt
            generator = self.anthropic.messages.stream(
                **model_kwargs,
            )
        except Exception as e:  # TODO: add retry logic
            raise e

        with generator as stream:
            for chunk in stream.text_stream:
                # print(chunk)
                yield chunk.encode("utf8")


class Anthropic(Grammarless):
    """Represents an Anthropic model as exposed through their remote API.

    Note that because this uses a remote API endpoint without built-in guidance support
    there are some things we cannot do, like force the model to follow a pattern inside
    a chat role block.
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        echo=True,
        api_key=None,
        timeout=0.5,
        max_streaming_tokens=1000,
        compute_log_probs=False,
        **kwargs,
    ):
        """Build a new Anthropic model object that represents a model in a given state."""

        super().__init__(
            engine=AnthropicEngine(
                model=model,
                tokenizer=tokenizer,
                api_key=api_key,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
                **kwargs,
            ),
            echo=echo,
        )
