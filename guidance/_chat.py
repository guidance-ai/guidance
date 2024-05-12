import abc
class ChatTemplate(abc.ABC):
    """Contains template for all chat and instruct tuned models."""

    @abc.abstractmethod
    def get_role_start(self, role_name, **kwargs):
        raise NotImplementedError(
            "You need to use a ChatTemplate subclass that overrides the get_role_start method"
        )

    @abc.abstractmethod
    def get_role_end(self, role_name=None):
        raise NotImplementedError(
            "You need to use a ChatTemplate subclass that overrides the get_role_start method"
        )


class UnsupportedRoleException(Exception):
    def __init__(self, role_name, instance):
        self.role_name = role_name
        self.instance = instance
        super().__init__(self._format_message())

    def _format_message(self):
        return (f"Role {self.role_name} is not supported by the {self.instance.__class__.__name__} chat template. "
                f"Use one of the following roles: {self.instance.available_roles} or pass in a new chat template.")


CHAT_TEMPLATE_CACHE = {}

# Llama-2
# [05/08/24] https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/tokenizer_config.json#L12
llama2_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
class Llama2ChatTemplate(ChatTemplate):
    available_roles = ["system", "user", "assistant"]
    template_str = llama2_template

    def get_role_start(self, role_name):
        if role_name == "system":
            return "[INST] <<SYS>>\n"
        elif role_name == "user":
            return "<s>[INST]"
        elif role_name == "assistant":
            return " "
        else:
            return UnsupportedRoleException(role_name, self)
        
    def get_role_end(self, role_name=None):
        if role_name == "system":
            return "\n<</SYS>"
        elif role_name == "user":
            return " [/INST]"
        elif role_name == "assistant":
            return "</s>"
        else:
            return UnsupportedRoleException(role_name, self)

CHAT_TEMPLATE_CACHE[llama2_template] = Llama2ChatTemplate

# Llama-3
# Fortunately Llama-3's chat template is WAY better than Llama-2's...
# [05/08/24] https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json#L2053
llama3_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
class Llama3ChatTemplate(ChatTemplate):
    available_roles = ["system", "user", "assistant"]
    template_str = llama3_template

    def get_role_start(self, role_name):
        if role_name == "system":
            return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        elif role_name == "user":
            return "<|start_header_id|>user<|end_header_id>\n\n"
        elif role_name == "assistant":
            return "<|start_header_id|>assistant<|end_header_id>\n\n"
        else:
            return UnsupportedRoleException(role_name, self)
        
    def get_role_end(self, role_name=None):
        return "<|eot_id|>"
CHAT_TEMPLATE_CACHE[llama3_template] = Llama3ChatTemplate

# Phi-3
# [05/08/24] https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/tokenizer_config.json#L119
# Phi-3 doesn't support system roles
phi3_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"
class Phi3ChatTemplate(ChatTemplate):
    available_roles = ["user", "assistant"]
    template_str = phi3_template

    def get_role_start(self, role_name):
        if role_name == "user":
            return "<|user|>\n"
        elif role_name == "assistant":
            return "<|assistant|>\n"
        else:
            return UnsupportedRoleException(role_name, self)
        
    def get_role_end(self, role_name=None):
        return "<|end|>\n"
    
CHAT_TEMPLATE_CACHE[phi3_template] = Phi3ChatTemplate

# Mistral-7B-Instruct-v0.2
# [05/08/24] https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/tokenizer_config.json#L42
mistral_7b_instruct_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
class Mistral7BInstructChatTemplate(ChatTemplate):
    available_roles = ["user", "assistant"]
    template_str = mistral_7b_instruct_template

    def get_role_start(self, role_name):
        if role_name == "user":
            return "[INST] "
        elif role_name == "assistant":
            return " "
        else:
            return UnsupportedRoleException(role_name, self)
        
    def get_role_end(self, role_name=None):
        if role_name == "user":
            return " [/INST]"
        elif role_name == "assistant":
            return "</s>"
        else:
            return UnsupportedRoleException(role_name, self)
        
CHAT_TEMPLATE_CACHE[mistral_7b_instruct_template] = Mistral7BInstructChatTemplate