import os
from ._model import Chat, Instruct
from ._openai import OpenAIChatEngine, OpenAI, OpenAIInstructEngine, OpenAICompletionEngine, OpenAIEngine
from .transformers._transformers import TransformersTokenizer


class TogetherAI(OpenAI):
    def __init__(self, model, tokenizer=None, echo=True, api_key=None, max_streaming_tokens=1000, timeout=0.5, compute_log_probs=False, engine_class=None, **kwargs):
        '''
        Build a new TogetherAI model object that represents a model in a given state.
        '''

        tokenizer = TransformersTokenizer(model=model, tokenizer=tokenizer, ignore_bos_token=True)

        # Default base_url is the together.ai endpoint
        if not "base_url" in kwargs:
            kwargs["base_url"] = 'https://api.together.xyz'
        # TogetherAI uses TOGETHERAI_API_KEY env value instead of OPENAI_API_KEY
        # We pass explicitly to avoid OpenAI class complaining about a missing key
        if api_key is None:
            api_key = os.environ.get("TOGETHERAI_API_KEY", None)
        if api_key is None:
            raise Exception(
                "The api_key client option must be set either by passing api_key to the client or by setting the TOGETHERAI_API_KEY environment variable"
            )

        if engine_class is None:
            engine_map = {
                TogetherAICompletion: OpenAICompletionEngine,
                TogetherAIInstruct: TogetherAIInstructEngine,
                TogetherAIChat: OpenAIChatEngine,
                TogetherAI: OpenAICompletionEngine,
            }
            for k in engine_map:
                if issubclass(self.__class__, k):
                    engine_class = engine_map[k]
                    break

        super().__init__(
            model, tokenizer, echo, api_key, max_streaming_tokens, timeout, compute_log_probs, engine_class, **kwargs
        )

class TogetherAICompletion(TogetherAI):
    pass


class TogetherAIInstruct(TogetherAI, Instruct):
    """
    Utilizes transformers chat templates to apply a user role to the prompt
    """
    def __init__(self, model, tokenizer=None, echo=True, api_key=None, max_streaming_tokens=1000, timeout=0.5, compute_log_probs=False, engine_class=None, **kwargs):
        self.dummy_prompt = "$GUIDANCE_DUMMY_INPUT$"
        super().__init__(
            model, tokenizer, echo, api_key, max_streaming_tokens, timeout, compute_log_probs, engine_class, **kwargs
        )

    def get_template_prompt(self):
        dummy_messages = [
            {
                "role": "user",
                "content": self.dummy_prompt,
            }
        ]
        template_prompt = self.engine.tokenizer._orig_tokenizer.apply_chat_template(dummy_messages, tokenize=False)
        return template_prompt


    def get_role_start(self, name):
        template_prompt = self.get_template_prompt()
        start = template_prompt[:template_prompt.index(self.dummy_prompt)]
        return start
    
    def get_role_end(self, name):
        if name == "instruction":
            template_prompt = self.get_template_prompt()
            end = template_prompt[template_prompt.index(self.dummy_prompt) + len(self.dummy_prompt):]
            return end
        else:
            raise Exception(f"The TogetherAIInstruct model does not know about the {name} role type!")


class TogetherAIInstructEngine(OpenAIEngine):
    def _generator(self, prompt, temperature):
        self._reset_shared_data(prompt, temperature)

        try:
            generator = self.client.completions.create(
                model=self.model_name,
                prompt=prompt.decode("utf8"), 
                max_tokens=self.max_streaming_tokens, 
                n=1, 
                top_p=1.0, # TODO: this should be controllable like temp (from the grammar)
                temperature=temperature, 
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e

        for part in generator:
            if len(part.choices) > 0:
                chunk = part.choices[0].text or ""
            else:
                chunk = ""
            yield chunk.encode("utf8")


class TogetherAIChat(TogetherAI, Chat):
    pass