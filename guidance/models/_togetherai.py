import os
from ._model import Chat, Instruct
from ._openai import (
    OpenAI,
    OpenAIEngine,
)
from .transformers._transformers import TransformersTokenizer


class TogetherAI(OpenAI):
    def __init__(
        self,
        model,
        tokenizer=None,
        echo=True,
        api_key=None,
        max_streaming_tokens=1000,
        timeout=0.5,
        compute_log_probs=False,
        engine_class=None,
        **kwargs,
    ):
        """
        Build a new TogetherAI model object that represents a model in a given state.
        """

        tokenizer = TransformersTokenizer(
            model=model, tokenizer=tokenizer, ignore_bos_token=True
        )

        # Default base_url is the together.ai endpoint
        if not "base_url" in kwargs:
            kwargs["base_url"] = "https://api.together.xyz"
        # TogetherAI uses TOGETHERAI_API_KEY env value instead of OPENAI_API_KEY
        # We pass explicitly to avoid OpenAI class complaining about a missing key
        if api_key is None:
            api_key = os.environ.get("TOGETHERAI_API_KEY", None)
        if api_key is None:
            raise Exception(
                "The api_key client option must be set either by passing api_key to the client or by setting the TOGETHERAI_API_KEY environment variable"
            )

        if engine_class is None:
            engine_map = {
                TogetherAICompletion: OpenAIEngine,
                TogetherAIInstruct: OpenAIEngine,
                TogetherAIChat: OpenAIEngine,
                TogetherAI: OpenAIEngine,
            }
            for k in engine_map:
                if issubclass(self.__class__, k):
                    engine_class = engine_map[k]
                    break

        super().__init__(
            model,
            tokenizer,
            echo,
            api_key,
            max_streaming_tokens,
            timeout,
            compute_log_probs,
            engine_class,
            **kwargs,
        )


class TogetherAICompletion(TogetherAI):
    pass


class TogetherAIInstruct(TogetherAI, Instruct):
    """
    Utilizes chat endpoints to simulate a single instruction query
    together.ai will format in correct prompt template for model on their end
    """

    def get_role_start(self, name):
        if name == "instruction":
            return "<|im_start|>user\n"
        else:
            raise Exception(
                f"The TogetherAIInstruct model does not know about the {name} role type!"
            )

    def get_role_end(self, name):
        if name == "instruction":
            return "<|im_end|>"
        else:
            raise Exception(
                f"The TogetherAIInstruct model does not know about the {name} role type!"
            )


class TogetherAIChat(TogetherAI, Chat):
    pass
