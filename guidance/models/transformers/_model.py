import re

from .._base import Model
from .._engine import EngineClient, EngineState, Llama3VisionClient, Phi3VisionClient
from ._engine import TransformersEngine


class Transformers(Model):
    def __init__(
        self,
        model=None,
        echo=True,
        compute_log_probs=False,
        chat_template=None,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        **kwargs,
    ):
        """Build a new Transformers model object that represents a model in a given state."""
        if re.search("Llama-3.*-Vision", model):
            client_cls = Llama3VisionClient
        elif re.search("Phi-3-vision", model):
            client_cls = Phi3VisionClient
        else:
            client_cls = EngineClient

        client = client_cls(
            True,
            TransformersEngine,
            model,
            compute_log_probs,
            chat_template=chat_template,
            enable_backtrack=enable_backtrack,
            enable_ff_tokens=enable_ff_tokens,
            enable_monitoring=enable_monitoring,
            **kwargs,
        )
        super().__init__(
            client=client,
            state=EngineState(),
            echo=echo,
        )
