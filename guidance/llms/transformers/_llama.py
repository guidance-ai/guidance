from .._transformers import Transformers

class LLaMA(Transformers):
    """ A HuggingFace transformers version of the LLaMA language model with Guidance support.
    """

    llm_name: str = "llama"

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):

        # load the LLaMA specific tokenizer and model
        import transformers
        if isinstance(model, str):
            if tokenizer is None:
                tokenizer = transformers.LlamaTokenizer.from_pretrained(model, **kwargs)
            model = transformers.LlamaForCausalLM.from_pretrained(model, **kwargs)
            
        return super()._model_and_tokenizer(model, tokenizer, **kwargs)