from ._llama import LLaMA

class Vicuna(LLaMA):
    """ A HuggingFace transformers version of the Vicuna language model with Guidance support.
    """

    llm_name: str = "vicuna"

    default_system_prompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

    @staticmethod
    def role_start(role):
        if role == 'user':
            return 'USER: '
        elif role == 'assistant':
            return 'ASSISTANT: '
        else:
            return ''
    
    @staticmethod
    def role_end(role):
        if role == 'user':
            return ''
        elif role == 'assistant':
            return '</s>'
        else:
            return ''