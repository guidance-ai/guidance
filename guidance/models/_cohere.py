from ._lite_llm import LiteLLM, LiteLLMCompletion, LiteLLMInstruct

class Cohere(LiteLLM):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, api_base=None, api_key=None, custom_llm_provider=None, temperature=0.0, max_streaming_tokens=1000, **kwargs):
        try:
            import tokenizers
        except ImportError:
            raise Exception("Please install the HuggingFace tokenizers package using `pip install tokenizers -U` in order to use guidance.models.Cohere!")

        # get the tokenizer
        if tokenizer is None:
            try:
                tokenizer = tokenizers.Tokenizer.from_pretrained("Cohere/"+model)
            except:
                tokenizer = tokenizers.Tokenizer.from_pretrained("Cohere/command-nightly")
        
        super().__init__(
            model, tokenizer=tokenizer, echo=echo,
            caching=caching, temperature=temperature,
            max_streaming_tokens=max_streaming_tokens, **kwargs
        )


class CohereCompletion(Cohere, LiteLLMCompletion):
    pass

class CohereInstruct(Cohere, LiteLLMInstruct):
    pass