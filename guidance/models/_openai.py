import re

import guidance.endpoints
from ._lm import LM, ChatLM


chat_model_pattern = r'^(gpt-3\.5-turbo|gpt-4)(-\d+k)?(-\d{4})?$'

class OpenAI(LM):
    def __init__(self, model, caching=True, **endpoint_kwargs):

        # subclass to ChatOpenAI if model is chat
        if re.match(chat_model_pattern, model) and self.__class__ is OpenAI:
            self.__class__ = ChatOpenAI
            ChatOpenAI.__init__(self, model=model, caching=caching)
            return

        # standard init
        super().__init__(model, caching=caching)
        self.model = model

        self.endpoint = guidance.endpoints.OpenAI(model, **endpoint_kwargs)
        self._endpoint_session = self.endpoint.session()

class ChatOpenAI(OpenAI, ChatLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tool_def(self, *args, **kwargs):
        lm = self + "<||_html:<span style='background-color: rgba(93, 63, 211, 0.15)'>_||>"
        lm = OpenAI.tool_def(lm, *args, **kwargs)
        return lm + "<||_html:</span>_||>"