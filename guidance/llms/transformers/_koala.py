from ._llama import LLaMA

class Koala(LLaMA):
    """ A HuggingFace transformers version of the Koala language model with Guidance support.
    """

    llm_name: str = "koala"

    default_system_prompt = """BEGINNING OF CONVERSATION:"""

    @staticmethod
    def role_start(role):
        if role == 'user':
            return 'USER: '
        elif role == 'assistant':
            return 'GPT: '
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