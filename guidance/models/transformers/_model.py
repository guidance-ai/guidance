import re

from .._base import Model
from .._engine import EngineClient, EngineState, Llama3VisionState, Phi3VisionState
from ._engine import TransformersEngine


class Transformers(Model):
    def __init__(
        self,
        model=None,
        tokenizer=None,
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
            state = Llama3VisionState()
        elif re.search("Phi-3-vision", model):
            state = Phi3VisionState()
        else:
            state = EngineState()

        client = EngineClient(
            TransformersEngine(
                model,
                tokenizer,
                compute_log_probs,
                chat_template=chat_template,
                enable_backtrack=enable_backtrack,
                enable_ff_tokens=enable_ff_tokens,
                enable_monitoring=enable_monitoring,
                **kwargs,
            )
        )
        super().__init__(
            client=client,
            state=state,
            echo=echo,
        )
