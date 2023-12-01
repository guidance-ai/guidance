import re
from .._model import Chat, Instruct
from .._remote import Remote


try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    import torch
    is_torch = True
except ImportError:
    is_torch = False

try:
    # TODO: can we eliminate the torch requirement for llama.cpp by using numpy in the caller instead?
    import vertexai
    is_vertexai = True
except ImportError:
    is_vertexai = False

class VertexAI(Remote):
    def __init__(self, model, tokenizer=None, echo=True, caching=True, temperature=0.0, top_p=1.0, max_streaming_tokens=None, **kwargs):
        if not is_vertexai:
            raise Exception("Please install the vertexai package using `pip install google-cloud-aiplatform` in order to use guidance.models.VertexAI!")
        
        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is VertexAI:
            found_subclass = None
            from .. import vertexai

            if isinstance(model, str):
                model_name = model
            else:
                model_name = self.model_obj._model_id # TODO: is this right?

            # CodeyCompletion
            if re.match("code-gecko(@[0-9]+)?", model_name):
                found_subclass = vertexai.CodeyCompletion

            # CodeyInstruct
            elif re.match("code-bison(@[0-9]+)?", model_name):
                found_subclass = vertexai.CodeyInstruct

            # CodeyChat
            elif re.match("codechat-bison(@[0-9]+)?", model_name):
                found_subclass = vertexai.CodeyChat

            # PaLM2Instruct
            elif re.match("text-(bison|unicorn)(@[0-9]+)?", model_name):
                found_subclass = vertexai.PaLM2Instruct

            # PaLM2Chat
            elif re.match("chat-bison(@[0-9]+)?", model_name):
                found_subclass = vertexai.PaLM2Chat
            
            # convert to any found subclass
            if found_subclass is not None:
                self.__class__ = found_subclass
                found_subclass.__init__(self, model, tokenizer=tokenizer, echo=echo, caching=caching, temperature=temperature, max_streaming_tokens=max_streaming_tokens, **kwargs)
                return # we return since we just ran init above and don't need to run again
        
            # make sure we have a valid model object
            if isinstance(model, str):
                raise Exception("The model ID you passed, `{model}`, does not match any known subclasses!")

        super().__init__(
            model, tokenizer=tokenizer, echo=echo,
            caching=caching, temperature=temperature, top_p=top_p,
            max_streaming_tokens=max_streaming_tokens, **kwargs
        )

class VertexAICompletion(VertexAI):

    def _generator(self, prompt):
        self._shared_state["not_running_stream"].clear() # so we know we are running
        self._shared_state["data"] = prompt # we start with this data

        try:
            kwargs = {}
            if self.max_streaming_tokens is not None:
                kwargs["max_output_tokens"] = self.max_streaming_tokens
            generator = self.model_obj.predict_streaming(
                prompt.decode("utf8"),
                #top_p=self.top_p,
                temperature=self.temperature,
                **kwargs
            )
        except Exception as e: # TODO: add retry logic
            raise e
        
        for chunk in generator:
            yield chunk.text.encode("utf8")

class VertexAIInstruct(VertexAI, Instruct):

    def get_role_start(self, name):
        return ""
    
    def get_role_end(self, name):
        if name == "instruction":
            return "<|endofprompt|>"
        else:
            raise Exception(f"The VertexAIInstruct model does not know about the {name} role type!")

    def _generator(self, prompt):
        # start the new stream
        prompt_end = prompt.find(b'<|endofprompt|>')
        if prompt_end >= 0:
            stripped_prompt = prompt[:prompt_end]
        else:
            raise Exception("This model cannot handle prompts that don't match the instruct format! Follow for example:\nwith instruction():\n    lm += prompt\nlm += gen(max_tokens=10)")
        self._shared_state["not_running_stream"].clear() # so we know we are running
        self._shared_state["data"] = stripped_prompt + b'<|endofprompt|>'# we start with this data
        kwargs = {}
        if self.max_streaming_tokens is not None:
            kwargs["max_output_tokens"] = self.max_streaming_tokens
        for chunk in self.model_obj.predict_streaming(self._shared_state["data"].decode("utf8"), **kwargs):
            yield chunk.text.encode("utf8")

class VertexAIChat(VertexAI, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generator(self, prompt):
        
        # find the system text
        pos = 0
        system_start = b'<|im_start|>system\n'
        user_start = b'<|im_start|>user\n'
        assistant_start = b'<|im_start|>assistant\n'
        role_end = b'<|im_end|>'
        # system_start_pos = prompt.startswith(system_start)
        
        # find the system text
        system_text = b''
        if prompt.startswith(system_start):
            pos += len(system_start)
            system_end_pos = prompt.find(role_end)
            system_text = prompt[pos:system_end_pos]
            pos = system_end_pos + len(role_end)

        # find the user/assistant pairs
        messages = []
        valid_end = False
        while True:

            # find the user text
            if prompt[pos:].startswith(user_start):
                pos += len(user_start)
                end_pos = prompt[pos:].find(role_end)
                if end_pos < 0:
                    break
                messages.append(vertexai.language_models.ChatMessage(
                    author="user",
                    content=prompt[pos:pos+end_pos].decode("utf8"),
                ))
                pos += end_pos + len(role_end)
            elif prompt[pos:].startswith(assistant_start):
                pos += len(assistant_start)
                end_pos = prompt[pos:].find(role_end)
                if end_pos < 0:
                    valid_end = True
                    break
                messages.append(vertexai.language_models.ChatMessage(
                    author="assistant",
                    content=prompt[pos:pos+end_pos].decode("utf8"),
                ))
                pos += end_pos + len(role_end)
            
        self._shared_state["data"] = prompt[:pos]

        assert len(messages) > 0, "Bad chat format! No chat blocks were defined."
        assert messages[-1].author == "user", "Bad chat format! There must be a user() role before the last assistant() role."
        assert valid_end, "Bad chat format! You must generate inside assistant() roles."

        # TODO: don't make a new session on every call
        last_user_text = messages.pop().content
        
        chat_session = self.model_obj.start_chat(
            context=system_text.decode("utf8"),
            message_history=messages,
        )

        kwargs = {}
        if self.max_streaming_tokens is not None:
            kwargs["max_output_tokens"] = self.max_streaming_tokens
        generator = chat_session.send_message_streaming(last_user_text, **kwargs)

        for chunk in generator:
            yield chunk.text.encode("utf8")