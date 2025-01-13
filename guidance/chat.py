import inspect
import warnings

from typing import Dict, Union


class ChatTemplate:
    """Contains template for all chat and instruct tuned models."""

    def get_role_start(self, role_name: str, **kwargs):
        raise NotImplementedError(
            "You need to use a ChatTemplate subclass that overrides the get_role_start method"
        )

    def get_role_end(self, role_name: Union[str, None] = None):
        raise NotImplementedError(
            "You need to use a ChatTemplate subclass that overrides the get_role_start method"
        )


class ChatTemplateCache:
    def __init__(self):
        self._cache: Dict[str, ChatTemplate] = {}

    def __getitem__(self, key: str) -> ChatTemplate:
        key_compact = key.replace(" ", "")
        return self._cache[key_compact]

    def __setitem__(self, key: str, value):
        key_compact = key.replace(" ", "")
        self._cache[key_compact] = value

    def __contains__(self, key: str):
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
        return f"Role {self.role_name} is not supported by the {self.instance.__class__.__name__} chat template. "


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
            raise Exception(
                "You can't use the base ChatTemplate class directly. Create or use a subclass instead."
            )
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
        warnings.warn(
            f"""Chat template {chat_template} was unable to be loaded directly into guidance.
                        Defaulting to the ChatML format which may not be optimal for the selected model. 
                        For best results, create and pass in a `guidance.ChatTemplate` subclass for your model."""
        )

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
            return "<s>[INST] "
        elif role_name == "assistant":
            return " "
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        if role_name == "system":
            return "\n<</SYS>>"
        elif role_name == "user":
            return " [/INST]"
        elif role_name == "assistant":
            return " </s>"
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
            return "<|start_header_id|>user<|end_header_id|>\n\n"
        elif role_name == "assistant":
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        return "<|eot_id|>"


CHAT_TEMPLATE_CACHE[llama3_template] = Llama3ChatTemplate

# --------------------------------------------------
# @@@@ Phi-3 @@@@
# --------------------------------------------------
# [05/08/24] https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/tokenizer_config.json#L119
phi3_mini_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}"
phi3_mini_llamacpp_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"


class Phi3MiniChatTemplate(ChatTemplate):
    # available_roles = ["user", "assistant"]
    template_str = phi3_mini_template

    def get_role_start(self, role_name):
        if role_name == "user":
            return "<|user|>\n"
        elif role_name == "assistant":
            return "<|assistant|>\n"
        elif role_name == "system":
            return "<|system|>\n"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        return "<|end|>\n"


CHAT_TEMPLATE_CACHE[phi3_mini_template] = Phi3MiniChatTemplate
CHAT_TEMPLATE_CACHE[phi3_mini_llamacpp_template] = Phi3MiniChatTemplate

# https://huggingface.co/microsoft/Phi-3-small-8k-instruct/blob/main/tokenizer_config.json
phi3_small_template = "{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}"


# https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/blob/main/tokenizer_config.json#L119
phi3_medium_template = "{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"


# Although the templates are different, the roles are the same between medium and small (for now)
class Phi3SmallMediumChatTemplate(ChatTemplate):
    # available_roles = ["user", "assistant"]
    template_str = phi3_small_template

    def get_role_start(self, role_name):
        if role_name == "user":
            return "<|user|>\n"
        elif role_name == "assistant":
            return "<|assistant|>\n"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        return "<|end|>\n"


CHAT_TEMPLATE_CACHE[phi3_small_template] = Phi3SmallMediumChatTemplate
CHAT_TEMPLATE_CACHE[phi3_medium_template] = Phi3SmallMediumChatTemplate

# --------------------------------------------------
# @@@@ Mistral-7B-Instruct-v0.2 @@@@
# --------------------------------------------------
# [05/08/24] https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/tokenizer_config.json#L42
mistral_7b_instruct_template = "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"
mistral_7b_instruct_llamacpp_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"


class Mistral7BInstructChatTemplate(ChatTemplate):
    # available_roles = ["user", "assistant"]
    template_str = mistral_7b_instruct_template

    def get_role_start(self, role_name):
        if role_name == "user":
            return " [INST] "
        elif role_name == "assistant":
            return " "
        elif role_name == "system":
            raise ValueError("Please include system instructions in the first user message")
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
CHAT_TEMPLATE_CACHE[mistral_7b_instruct_llamacpp_template] = Mistral7BInstructChatTemplate

# --------------------------------------------------
# @@@@ Gemma-2-9b-it @@@@
# --------------------------------------------------
# From https://huggingface.co/google/gemma-2-9b-it/blob/main/tokenizer_config.json#L1747
gemma2_9b_it_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"


class Gemma29BInstructChatTemplate(ChatTemplate):
    # available_roles = ["user", "assistant"]
    template_str = gemma2_9b_it_template

    def get_role_start(self, role_name):
        if role_name == "user":
            return "<start_of_turn>user\n"
        elif role_name == "assistant":
            return "<start_of_turn>model\n"
        elif role_name == "system":
            raise ValueError("System Role not supported")
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        if role_name == "user":
            return "<end_of_turn>\n"
        elif role_name == "assistant":
            return "<end_of_turn>\n"
        else:
            raise UnsupportedRoleException(role_name, self)


CHAT_TEMPLATE_CACHE[gemma2_9b_it_template] = Gemma29BInstructChatTemplate

# --------------------------------------------------
# @@@@ Qwen2.5 @@@@
# --------------------------------------------------
# From https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/tokenizer_config.json#L198
qwen2dot5_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"

# From https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer_config.json#L198
qwen2dot5_it_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"


class Qwen2dot5ChatTemplate(ChatTemplate):
    # available_roles = ["system", "user", "assistant"]
    template_str = qwen2dot5_it_template

    def get_role_start(self, role_name):
        if role_name in ["system", "user", "assistant"]:
            return f"<|im_start|>{role_name}\n"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        return "<|im_end|>\n"


CHAT_TEMPLATE_CACHE[qwen2dot5_template] = Qwen2dot5ChatTemplate
CHAT_TEMPLATE_CACHE[qwen2dot5_it_template] = Qwen2dot5ChatTemplate


# TODO WIP [HN]: Llama 3.2 has the same issues as Phi-3 mini, where trim behavior is defined in the template.
# Additionally, they use some crazy jinja trick to dynamically pull the current date (but have a fallback of July 26, 2024).
# There's an additional problem that system roles are ALWAYS defined. Might need a refactor the entire system to handle.
class Llama3dot2ChatTemplate(ChatTemplate):
    # available_roles = ["user", "assistant"]
    template_str = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'

    def get_role_start(self, role_name):
        if role_name == "system":
            # TODO: Figure out how Jinja detects datetime and replace with strftime() if needed - currently using fallback
            return "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"
        elif role_name == "user":
            return "<|start_header_id|>user<|end_header_id|>\n\n"
        elif role_name == "assistant":
            return "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        return "<|eot_id|>"