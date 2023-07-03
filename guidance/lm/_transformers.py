import re

from ._lm import LM
from ._chat_lm import ChatLM
from guidance.llms import Transformers as TransformersEndpoint
from guidance.llms import SyncSession


chat_model_pattern = r'^(gpt-3\.5-turbo|gpt-4)(-\d+k)?(-\d{4})?$'

class Transformers(LM):
    def __init__(self, model, **kwargs):

        # subclass to ChatTransformers if model is chat
        if re.match(chat_model_pattern, model) and self.__class__ is Transformers:
            self.__class__ = ChatTransformers
            ChatTransformers.__init__(self, model=model, **kwargs)
            return

        # standard init
        super().__init__(model, **kwargs)
        self.model = model

        self.endpoint = TransformersEndpoint(model, **kwargs)
        self.endpoint.caching = True
        self.session = SyncSession(self.endpoint.session())

class ChatTransformers(Transformers, ChatLM):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)