import warnings
import uuid
import inspect

class ChatTemplate:
    """Contains template for all chat and instruct tuned models."""

    def get_role_start(self, role_name, **kwargs):
        raise NotImplementedError(
            "You need to use a ChatTemplate subclass that overrides the get_role_start method"
        )

    def get_role_end(self, role_name=None):
        raise NotImplementedError(
            "You need to use a ChatTemplate subclass that overrides the get_role_start method"
        )
    
class ChatTemplateCache: 
    def __init__(self):
        self._cache = {}

    def __getitem__(self, key):
        key_compact = key.replace(" ", "")
        return self._cache[key_compact]


    def __setitem__(self, key, value):
        key_compact = key.replace(" ", "")
        self._cache[key_compact] = value

    def __contains__(self, key):
        key_compact = key.replace(" ", "")
        return key_compact in self._cache
    
# Feels weird having to instantiate this, but it's a singleton for all purposes
# TODO [HN]: Add an alias system so we can instantiate with other simple keys (e.g. "llama2" instead of the full template string)
CHAT_TEMPLATE_CACHE = ChatTemplateCache() 

class UnsupportedRoleException(Exception):
    def __init__(self, role_name, instance):
        self.role_name = role_name
        self.instance = instance
        super().__init__(self._format_message())

    def _format_message(self):
        return (f"Role {self.role_name} is not supported by the {self.instance.__class__.__name__} chat template. ")

def load_template_class(chat_template=None):
    """Utility method to find the best chat template.
    
    Order of precedence:
    - If it's a chat template class, use it directly
    - If it's a string, check the cache of popular model templates
    - If it's a string and not in the cache, try to create a class dynamically
    - [TODO] If it's a string and can't be created, default to ChatML and raise a warning
    - If it's None, default to ChatML and raise a warning
    """
    if inspect.isclass(chat_template) and issubclass(chat_template, ChatTemplate):
        if chat_template is ChatTemplate:
            raise Exception("You can't use the base ChatTemplate class directly. Create or use a subclass instead.")
        return chat_template
    
    elif isinstance(chat_template, str):
        # First check the cache of popular model types
        # TODO: Expand keys of cache to include aliases for popular model types (e.g. "llama2, phi3")
        # Can possibly accomplish this with an "aliases" dictionary that maps all aliases to the canonical key in cache
        if chat_template in CHAT_TEMPLATE_CACHE:
            return CHAT_TEMPLATE_CACHE[chat_template]
        # TODO: Add logic here to try to auto-create class dynamically via _template_class_from_string method
    
    # Only warn when a user provided a chat template that we couldn't load
    if chat_template is not None:
        warnings.warn(f"""Chat template {chat_template} was unable to be loaded directly into guidance.
                        Defaulting to the ChatML format which may not be optimal for the selected model. 
                        For best results, create and pass in a `guidance.ChatTemplate` subclass for your model.""")
    
    # By default, use the ChatML Template. Warnings to user will happen downstream only if they use chat roles.
    return ChatMLTemplate
        

def _template_class_from_string(template_str):
    """Utility method to try to create a chat template class from a string."""
    # TODO: Try to build this, perhaps based on passing unit tests we create?
    pass


# CACHE IMPLEMENTATIONS:

# --------------------------------------------------
# @@@@ ChatML @@@@
# --------------------------------------------------
# Note that all grammarless models will default to this syntax, since we typically send chat formatted messages.
chatml_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
class ChatMLTemplate(ChatTemplate):
    template_str = chatml_template

    def get_role_start(self, role_name):
        return f"<|im_start|>{role_name}\n"
        
    def get_role_end(self, role_name=None):
        return "<|im_end|>\n"
    
CHAT_TEMPLATE_CACHE[chatml_template] = ChatMLTemplate


# --------------------------------------------------
# @@@@ Llama-2 @@@@
# --------------------------------------------------
# [05/08/24] https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/tokenizer_config.json#L12
llama2_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
class Llama2ChatTemplate(ChatTemplate):
    # available_roles = ["system", "user", "assistant"]
    template_str = llama2_template

    def get_role_start(self, role_name):
        if role_name == "system":
            return "[INST] <<SYS>>\n"
        elif role_name == "user":
            return "<s>[INST]"
        elif role_name == "assistant":
            return " "
        else:
            raise UnsupportedRoleException(role_name, self)
        
    def get_role_end(self, role_name=None):
        if role_name == "system":
            return "\n<</SYS>"
        elif role_name == "user":
            return " [/INST]"
        elif role_name == "assistant":
            return "</s>"
        else:
            raise UnsupportedRoleException(role_name, self)

CHAT_TEMPLATE_CACHE[llama2_template] = Llama2ChatTemplate


# --------------------------------------------------
# @@@@ Llama-3 @@@@
# --------------------------------------------------
# [05/08/24] https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json#L2053
llama3_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
class Llama3ChatTemplate(ChatTemplate):
    # available_roles = ["system", "user", "assistant"]
    template_str = llama3_template

    def get_role_start(self, role_name):
        if role_name == "system":
            return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        elif role_name == "user":
            return "<|start_header_id|>user<|end_header_id>\n\n"
        elif role_name == "assistant":
            return "<|start_header_id|>assistant<|end_header_id>\n\n"
        else:
            raise UnsupportedRoleException(role_name, self)
        
    def get_role_end(self, role_name=None):
        return "<|eot_id|>"
    
CHAT_TEMPLATE_CACHE[llama3_template] = Llama3ChatTemplate

# --------------------------------------------------
# @@@@ Phi-3 @@@@
# --------------------------------------------------
# [05/08/24] https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/tokenizer_config.json#L119
phi3_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"
class Phi3ChatTemplate(ChatTemplate):
    # available_roles = ["user", "assistant"]
    template_str = phi3_template

    def get_role_start(self, role_name):
        if role_name == "user":
            return "<|user|>"
        elif role_name == "assistant":
            return "<|assistant|>"
        else:
            raise UnsupportedRoleException(role_name, self)
        
    def get_role_end(self, role_name=None):
        return "<|end|>"
    
CHAT_TEMPLATE_CACHE[phi3_template] = Phi3ChatTemplate


# --------------------------------------------------
# @@@@ Mistral-7B-Instruct-v0.2 @@@@
# --------------------------------------------------
# [05/08/24] https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/tokenizer_config.json#L42
mistral_7b_instruct_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
class Mistral7BInstructChatTemplate(ChatTemplate):
    # available_roles = ["user", "assistant"]
    template_str = mistral_7b_instruct_template

    def get_role_start(self, role_name):
        if role_name == "user":
            return "[INST] "
        elif role_name == "assistant":
            return ""
        else:
            raise UnsupportedRoleException(role_name, self)
        
    def get_role_end(self, role_name=None):
        if role_name == "user":
            return " [/INST]"
        elif role_name == "assistant":
            return "</s>"
        else:
            raise UnsupportedRoleException(role_name, self)
        
CHAT_TEMPLATE_CACHE[mistral_7b_instruct_template] = Mistral7BInstructChatTemplate