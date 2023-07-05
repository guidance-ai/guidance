import re

import guidance.endpoints
from ._lm import LM, ChatLM


chat_model_pattern = r'^(gpt-3\.5-turbo|gpt-4)(-\d+k)?(-\d{4})?$'

class OpenAI(LM):
    def __init__(self, model, **kwargs):

        # subclass to ChatOpenAI if model is chat
        if re.match(chat_model_pattern, model) and self.__class__ is OpenAI:
            self.__class__ = ChatOpenAI
            ChatOpenAI.__init__(self, model=model, **kwargs)
            return

        # standard init
        super().__init__(model, **kwargs)
        self.model = model

        self.endpoint = guidance.endpoints.OpenAI(model, **kwargs)
        self._endpoint_session = self.endpoint.session()

class ChatOpenAI(OpenAI, ChatLM):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)