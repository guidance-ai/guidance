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

class LLaMAChat(LLaMA):

    default_system_prompt = """You are a helpful assistant."""

    # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L45
    # This is not exatly right, I think they do a single [INST] for the system prompt AND first instruction
    @staticmethod
    def role_start(role):
       return  {
        'system': '[INST] <<SYS>>\n',
        'user': '[INST] ',
        'assistant': '',
        }[role]
    
    @staticmethod
    def role_end(role):
       return  {
        'system': '\n<</SYS>>\n\n[/INST]',
        'user': '[/INST]',
        'assistant': '',
        }[role]