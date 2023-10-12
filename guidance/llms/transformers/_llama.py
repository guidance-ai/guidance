from .._transformers import Transformers


class LLaMA(Transformers):
    """A HuggingFace transformers version of the LLaMA language model with Guidance support."""

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
    default_system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    _B_INST, _E_INST = "[INST] ", " [/INST]"
    _B_SYS, _E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    _B_SEQ, _E_SEQ = "<s>", "</s>"
    _dangling_inst = False
    _sequence_ended = True

    @staticmethod
    def role_start(role):
        _output = ""

        if LLaMAChat._sequence_ended:
            _output += LLaMAChat._B_SEQ

        if role == "system":
            LLaMAChat._dangling_inst = True
            return f"{_output}{LLaMAChat._B_INST}{LLaMAChat._B_SYS}"
        elif role == "user":
            return "" if LLaMAChat._dangling_inst else f"{_output}{LLaMAChat._B_INST}"
        elif role == "assistant":
            pass

    @staticmethod
    def role_end(role):
        if role == "system":
            return LLaMAChat._E_SYS
        elif role == "user":
            if LLaMAChat._dangling_inst:
                LLaMAChat._dangling_inst = False
            return LLaMAChat._E_INST
        elif role == "assistant":
            LLaMAChat._dangling_inst = False
            LLaMAChat._sequence_ended = True
            return LLaMAChat._E_SEQ
