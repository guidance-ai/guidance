from ._lite_llm import LiteLLMEngine, LiteLLM, LiteLLMCompletion, LiteLLMInstruct


class Cohere(LiteLLM):
    def __init__(
        self,
        model,
        tokenizer=None,
        echo=True,
        timeout=0.5,
        compute_log_probs=False,
        max_streaming_tokens=1000,
    ):
        """Build a new Anthropic model object that represents a model in a given state."""
        try:
            import tokenizers
        except ModuleNotFoundError:
            raise Exception(
                "Please install the HuggingFace tokenizers package using `pip install tokenizers -U` in order to use guidance.models.Cohere!"
            )

        # get the tokenizer
        if tokenizer is None:
            try:
                tokenizer = tokenizers.Tokenizer.from_pretrained("Cohere/" + model)
            except:
                tokenizer = tokenizers.Tokenizer.from_pretrained(
                    "Cohere/command-nightly"
                )

        super().__init__(
            model,
            tokenizer=tokenizer,
            echo=echo,
            timeout=timeout,
            max_streaming_tokens=max_streaming_tokens,
            compute_log_probs=compute_log_probs,
        )


class CohereCompletion(Cohere, LiteLLMCompletion):
    pass


class CohereInstruct(Cohere, LiteLLMInstruct):
    pass
