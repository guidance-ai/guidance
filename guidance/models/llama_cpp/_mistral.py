from .._model import Chat, Instruct
from ._llama_cpp import LlamaCpp , LlamaCppChat

class MistralChat(LlamaCpp, Chat):
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


class MistralChatOpenOrca(LlamaCppChat, Chat):
    def get_role_start(self, role_name, **kwargs):
        if role_name == "user":
            return "<|im_start|>user\n"
        
        elif role_name == "assistant":
            return "<|im_start|>assistant\n"
        
        elif role_name == "system":
            return "<|im_start|>system\n"            
    
    def get_role_end(self, role_name=None):
        return "<|im_end|>\n"
