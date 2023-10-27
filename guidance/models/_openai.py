import re
import os

from ._model import Model, Chat


chat_model_pattern = r'^(gpt-3\.5-turbo|gpt-4)(-\d+k)?(-\d{4})?$'

class OpenAI(Model):
    def __init__(self, model, api_key=None, organization=None, base_url=None, echo=True, eos_token="<|endoftext|>"):
        '''Initialize an OpenAI model. Parameters are based on the new OpenAI V1 SDK.

        Args:
            model: Supported OpenAI model name.
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY environment variable.
            organization: OpenAI organization ID. Defaults to OPENAI_ORG_ID environment variable.
            base_url: OpenAI API base URL. Defaults to "https://api.openai.com/v1".
            echo: Whether to echo the prompt back to the output. Defaults to True.
        '''
        # TODO: Re-enable caching, retry, throttling support.

        # subclass to OpenAIChat if model is chat
        if re.match(chat_model_pattern, model) and self.__class__ is OpenAI:
            # Will enable/test Chat support after getting basic completion models working.
            raise Exception("OpenAI chat models are not yet supported. Use gpt-3.5-instruct-turbo for now.") 
            self.__class__ = OpenAIChat
            OpenAIChat.__init__(self, model=model, caching=caching)
            return
        
        # standard init
        super().__init__(model)
        self.model_name = model

        # Configure an AsyncOpenAI Client with user params.
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if organization is None:
            organization = os.environ.get("OPENAI_ORG_ID")

        if base_url is None:
            base_url = r"https://api.openai.com/v1"

        try:
            from openai import OpenAI as OpenAIClient # Need to avoid naming conflict
            import tiktoken
        except ImportError:
            raise ImportError("OpenAI support requires openai >= 1 and tiktoken to be installed.")
        
        self.tokenizer = tiktoken.encoding_for_model(model)
        if isinstance(eos_token, str):
            self.eos_token = bytes(eos_token, "utf-8")
        else:
            self.eos_token = b"<|endoftext|>"
        
        eos_token_id = self.tokenizer.encode(self.eos_token.decode(), allowed_special={self.eos_token.decode()})
        assert len(eos_token_id) == 1 and isinstance(eos_token_id[0], int), "Could not find a single token representation of EOS token."
        self.eos_token_id = eos_token_id[0]


        self.client = OpenAIClient(api_key=api_key, organization=organization, base_url=base_url)

    def _openai_completion_call(self, prompt, max_tokens=100, n=1, top_p=1, temperature=0.0):
        try:
            return self.client.completions.create(
                model=self.model_name, 
                prompt=prompt, 
                max_tokens=max_tokens, 
                n=n, 
                top_p=top_p, 
                temperature=temperature, 
                stream=True
            )
        except Exception as e: # TODO: add retry logic
            raise e
        

    def __call__(self, grammar, max_tokens=100, n=1, top_p=1, temperature=0.0):
        # TODO: Validate the grammar (assuming perfectly valid gen for now)

        # TODO: Add support for gen intermixed with generation (i.e. grammar has a fix)
        
        # Make API call assuming the grammar is correct right now
        for completion in self._openai_completion_call(str(self), max_tokens, n, top_p, temperature):
            
            # Check if we are done
            completion_text = completion.choices[0].text
            if completion_text == self.eos_token:
                break # Finish if we hit our stop token -- TODO: update this to split out the stop properly from the grammar

            # Otherwise, continue to yield out bytes
            yield bytes(completion.choices[0].text, "utf-8"), True, 0.0, {}, {} # TODO: set logprobs if we have them

        # Outline
        # grammar_type = "gen"

        # if grammar_type == "gen":
        #     # Run a stream here and check for the stop condition of the grammar.
        #     pass

        # elif grammar_type == "select":
        #     # Port over the select logic from the old guidance code
        #     pass

        # else:
        #     raise Exception("OpenAI models are only able to support basic generation and selection grammars.")


class OpenAIChat(OpenAI, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tool_def(self, *args, **kwargs):
        lm = self + "<||_html:<span style='background-color: rgba(93, 63, 211, 0.15)'>_||>"
        lm = OpenAI.tool_def(lm, *args, **kwargs)
        return lm + "<||_html:</span>_||>"