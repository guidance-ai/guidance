import re

import guidance.endpoints
from ._lm import LM, ChatLM


class Transformers(LM):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.model = model

        self.endpoint = guidance.endpoints.Transformers(model, **kwargs)
        self._endpoint_session = self.endpoint.session()

class ChatTransformers(Transformers, ChatLM):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)