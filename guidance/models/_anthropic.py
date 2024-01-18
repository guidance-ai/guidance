import os
import tiktoken

from ._model import Chat, Instruct
from ._grammarless import GrammarlessEngine, Grammarless

class AnthropicEngine(GrammarlessEngine):
    def __init__(self, model, tokenizer, api_key, timeout, max_streaming_tokens, compute_log_probs, **kwargs):        
        try:
            from anthropic import Anthropic
        except ImportError:
            raise Exception("Please install the anthropic package version >= 0.7 using `pip install anthropic -U` in order to use guidance.models.Anthropic!")
        
        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is Anthropic:
            raise Exception("The Anthropic class is not meant to be used directly! Please use AnthropicChat assuming the model you are using is chat-based.")

        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if api_key is None:
            raise Exception("Expected an api_key argument or the ANTHROPIC_API_KEY environment variable to be set!")

        self.anthropic = Anthropic(api_key=api_key, **kwargs)

        self.model_name = model

        # we pretend it tokenizes like gpt2 if tiktoken does not know about it... TODO: make this better
        if tokenizer is None:
            try:
                tokenizer = tiktoken.encoding_for_model(model)
            except:
                tokenizer = tiktoken.get_encoding("gpt2")

        super().__init__(tokenizer, max_streaming_tokens, timeout, compute_log_probs)

    def _generator(self, prompt, temperature):

        # update our shared data state
        self._reset_shared_data(prompt, temperature)

        try:
            generator = self.anthropic.completions.create(
                model=self.model_name,
                prompt=prompt.decode("utf8"),
                max_tokens_to_sample=self.max_streaming_tokens,
                stream=True,
                temperature=temperature
            )
        except Exception as e: # TODO: add retry logic
            raise e
        
        for part in generator:
            chunk = part.completion or ""
            # print(chunk)
            yield chunk.encode("utf8")

class Anthropic(Grammarless):
    '''Represents an Anthropic model as exposed through their remote API.
    
    Note that because this uses a remote API endpoint without built-in guidance support
    there are some things we cannot do, like force the model to follow a pattern inside
    a chat role block.
    '''
    def __init__(self, model, tokenizer=None, echo=True, api_key=None, timeout=0.5, max_streaming_tokens=1000, compute_log_probs=False, **kwargs):
        '''Build a new Anthropic model object that represents a model in a given state.'''
        
        # if we are called directly (as opposed to through super()) then we convert ourselves to a more specific subclass if possible
        if self.__class__ is Anthropic:
            found_subclass = None

            # chat
            found_subclass = AnthropicChat # we assume all models are chat right now
            
            # convert to any found subclass
            self.__class__ = found_subclass
            found_subclass.__init__(self, model, tokenizer=tokenizer, echo=echo, api_key=api_key, max_streaming_tokens=max_streaming_tokens, timeout=timeout, compute_log_probs=compute_log_probs, **kwargs)
            return # we return since we just ran init above and don't need to run again

        super().__init__(
            engine=AnthropicEngine(
                model=model,
                tokenizer=tokenizer,
                api_key=api_key,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
                **kwargs
            ),
            echo=echo
        )

class AnthropicChat(Anthropic, Chat):
    def get_role_start(self, role_name, **kwargs):
        if role_name == "user":
            return "\n\nHuman:"
        if role_name == "assistant":
            return "\n\nAssistant:"
        if role_name == "system":
            return ""
    
    def get_role_end(self, role_name=None):
        return ""
    

