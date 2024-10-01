import logging
from ._model import Engine, Model
from ._tokenizer import Tokenizer

logger = logging.getLogger(__name__)

class LocalEngine(Engine):
    def __init__(self, tokenizer: Tokenizer, compute_log_probs=False):
        super().__init__(tokenizer, compute_log_probs)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class LocalModel(Model):
    def __init__(self, engine, echo=True, **kwargs):
        """Build a new Local model object that represents a model in a given state."""

        super().__init__(engine, echo, **kwargs)
