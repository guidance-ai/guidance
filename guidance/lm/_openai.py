import re

from guidance.llms import OpenAI as OpenAIEndpoint, SyncSession
from ._lm import LM
from ._chat_lm import ChatLM


chat_model_pattern = r'^(gpt-3\.5-turbo|gpt-4)(-\d+k)?(-\d{4})?$'

class OpenAI(LM):
    def __init__(self, model, caching=True, **kwargs):

        # subclass to ChatOpenAI if model is chat
        if re.match(chat_model_pattern, model) and self.__class__ is OpenAI:
            self.__class__ = ChatOpenAI
            ChatOpenAI.__init__(self, model=model, caching=caching, **kwargs)
            return

        # standard init
        super().__init__(model, **kwargs)
        self.model = model

        self.endpoint = OpenAIEndpoint(model, **kwargs)
        self.endpoint.caching = caching
        self.session = SyncSession(self.endpoint.session())

class ChatOpenAI(OpenAI, ChatLM):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)