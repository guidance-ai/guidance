from .._transformers import Transformers


class Falcon(Transformers):
    """A HuggingFace transformers version of the tiiuae Falcon language model with Guidance support."""

    llm_name: str = "falcon"

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):
        import transformers

        if isinstance(model, str):
            if tokenizer is None:
                tokenizer = transformers.AutoTokenizer.from_pretrained(model, **kwargs)

            model = transformers.AutoModelForCausalLM.from_pretrained(
                model, trust_remote_code=True, **kwargs
            )

        return super()._model_and_tokenizer(model, tokenizer, **kwargs)


class FalconChat(Falcon):
    default_system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    @staticmethod
    def role_start(role):
        return {
            "user": "User:\n",
            "system": "System:\n",
            "assistant": "Assistant:\n",
        }[role]

    @staticmethod
    def role_end(role):
        return ""
