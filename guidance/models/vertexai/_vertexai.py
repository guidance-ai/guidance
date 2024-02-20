import re
from .._model import Chat, Instruct
from .._grammarless import GrammarlessEngine, Grammarless

try:
    import vertexai
    is_vertexai = True
except ImportError:
    is_vertexai = False

class VertexAIEngine(GrammarlessEngine):
    def __init__(self, tokenizer, max_streaming_tokens, timeout, compute_log_probs, model_obj):
        super().__init__(tokenizer, max_streaming_tokens, timeout, compute_log_probs)
        self.model_obj = model_obj

class VertexAI(Grammarless):
    def __init__(self, model, tokenizer=None, echo=True, max_streaming_tokens=None, timeout=0.5, compute_log_probs=False, engine_class=None, **kwargs):
        '''Build a new VertexAI model object that represents a model in a given state.'''
        if not is_vertexai:
            raise Exception("Please install the vertexai package using `pip install google-cloud-aiplatform` in order to use guidance.models.VertexAI!")
        
        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is VertexAI:
            found_subclass = None
            from .. import vertexai as vertexai_subclasses

            if isinstance(model, str):
                model_name = model
            else:
                model_name = model._model_id

            # CodeyCompletion
            if re.match("code-gecko(@[0-9]+)?", model_name):
                found_subclass = vertexai_subclasses.CodeyCompletion

            # CodeyInstruct
            elif re.match("code-bison(@[0-9]+)?", model_name):
                found_subclass = vertexai_subclasses.CodeyInstruct

            # CodeyChat
            elif re.match("codechat-bison(@[0-9]+)?", model_name):
                found_subclass = vertexai_subclasses.CodeyChat

            # PaLM2Instruct
            elif re.match("text-(bison|unicorn)(@[0-9]+)?", model_name):
                found_subclass = vertexai_subclasses.PaLM2Instruct

            # PaLM2Chat
            elif re.match("chat-bison(@[0-9]+)?", model_name):
                found_subclass = vertexai_subclasses.PaLM2Chat

            # Gemini2Chat
            elif re.match("gemini-pro(@[0-9]+)?", model_name):
                found_subclass = vertexai_subclasses.GeminiChat
            
            # convert to any found subclass
            if found_subclass is not None:
                self.__class__ = found_subclass
                found_subclass.__init__(self, model, tokenizer=tokenizer, echo=echo, max_streaming_tokens=max_streaming_tokens, **kwargs)
                return # we return since we just ran init above and don't need to run again
        
            # make sure we have a valid model object
            if isinstance(model, str):
                raise Exception("The model ID you passed, `{model}`, does not match any known subclasses!")

        # this allows us to use a single constructor for all our subclasses
        if engine_class is None:
            engine_map = {
                VertexAICompletion: VertexAICompletionEngine,
                VertexAIInstruct: VertexAIInstructEngine,
                VertexAIChat: VertexAIChatEngine
            }
            for k in engine_map:
                if issubclass(self.__class__, k):
                    engine_class = engine_map[k]
                    break

        super().__init__(
            engine_class(tokenizer=tokenizer, timeout=timeout, compute_log_probs=compute_log_probs, max_streaming_tokens=max_streaming_tokens, model_obj=model),
            echo=echo
        )

class VertexAICompletion(VertexAI):
    pass

class VertexAICompletionEngine(VertexAIEngine):

    def _generator(self, prompt, temperature):
        self._not_running_stream.clear() # so we know we are running
        self._data = prompt # we start with this data

        try:
            kwargs = {}
            if self.max_streaming_tokens is not None:
                kwargs["max_output_tokens"] = self.max_streaming_tokens
            generator = self.model_obj.predict_streaming(
                prompt.decode("utf8"),
                #top_p=self.top_p,
                temperature=temperature,
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

class VertexAIInstructEngine(VertexAIEngine):
    def _generator(self, prompt, temperature):
        # start the new stream
        prompt_end = prompt.find(b'<|endofprompt|>')
        if prompt_end >= 0:
            stripped_prompt = prompt[:prompt_end]
        else:
            raise Exception("This model cannot handle prompts that don't match the instruct format! Follow for example:\nwith instruction():\n    lm += prompt\nlm += gen(max_tokens=10)")
        self._not_running_stream.clear() # so we know we are running
        self._data = stripped_prompt + b'<|endofprompt|>'# we start with this data
        kwargs = {}
        if self.max_streaming_tokens is not None:
            kwargs["max_output_tokens"] = self.max_streaming_tokens
        for chunk in self.model_obj.predict_streaming(self._data.decode("utf8"), temperature=temperature, **kwargs):
            yield chunk.text.encode("utf8")

class VertexAIChat(VertexAI, Chat):
    pass
class VertexAIChatEngine(VertexAIEngine):

    def _generator(self, prompt, temperature):
        
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
                messages.append(dict(
                    role="user",
                    content=prompt[pos:pos+end_pos].decode("utf8"),
                ))
                pos += end_pos + len(role_end)
            elif prompt[pos:].startswith(assistant_start):
                pos += len(assistant_start)
                end_pos = prompt[pos:].find(role_end)
                if end_pos < 0:
                    valid_end = True
                    break
                messages.append(dict(
                    role="assistant",
                    content=prompt[pos:pos+end_pos].decode("utf8"),
                ))
                pos += end_pos + len(role_end)
            else:
                raise Exception("It looks like your prompt is not a well formed chat prompt! Please enclose all model state appends inside chat role blocks like `user()` or `assistant()`.")
            
        self._data = prompt[:pos]

        assert len(messages) > 0, "Bad chat format! No chat blocks were defined."
        assert messages[-1]["role"] == "user", "Bad chat format! There must be a user() role before the last assistant() role."
        assert valid_end, "Bad chat format! You must generate inside assistant() roles."

        # TODO: don't make a new session on every call
        # last_user_text = messages.pop().content
        
        return self._start_generator(system_text.decode("utf8"), messages, temperature)

        # kwargs = {}
        # if self.max_streaming_tokens is not None:
        #     kwargs["max_output_tokens"] = self.max_streaming_tokens
        # generator = chat_session.send_message_streaming(last_user_text, temperature=temperature, **kwargs)

        # for chunk in generator:
        #     yield chunk.text.encode("utf8")

    def _start_generator(self, system_text, messages, temperature):
        messages = [vertexai.language_models.ChatMessage(author=m["role"], content=m["content"]) for m in messages]
        last_user_text = messages.pop().content

        chat_session = self.model_obj.start_chat(
            context=system_text,
            message_history=messages,
        )

        kwargs = {}
        if self.max_streaming_tokens is not None:
            kwargs["max_output_tokens"] = self.max_streaming_tokens
        generator = chat_session.send_message_streaming(last_user_text, temperature=temperature, **kwargs)

        for chunk in generator:
            yield chunk.text.encode("utf8")