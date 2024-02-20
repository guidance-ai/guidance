from .._model import Instruct
from ._llama_cpp import LlamaCpp, LlamaCppChat

class MistralChat(LlamaCppChat):
    def get_role_start(self, role_name, **kwargs):
        if role_name == "user":
            return "[INST] "
        
        elif role_name == "assistant":
            return ""
        
        elif role_name == "system":
            raise Exception("MistralChat does not support a sytem role!")
    
    def get_role_end(self, role_name=None):
        if role_name == "user":
            return " [/INST]"
        elif role_name == "assistant":
            return "</s>"
        elif role_name == "system":
            raise Exception("MistralChat does not support a sytem role!")
        
class MistralInstruct(LlamaCpp, Instruct):
    def get_role_start(self, role_name, **kwargs):
        if role_name == "instruction":
            return "[INST] "
    
    def get_role_end(self, role_name=None):
        if role_name == "instruction":
            return " [/INST]"