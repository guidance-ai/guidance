from .._engine import ModelWithEngine
from ._engine import TransformersEngine


# TODO: Expose a non-chat version
class Transformers(ModelWithEngine):
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
        super().__init__(
            TransformersEngine(
                model,
                tokenizer,
                compute_log_probs,
                chat_template=chat_template,
                enable_backtrack=enable_backtrack,
                enable_ff_tokens=enable_ff_tokens,
                enable_monitoring=enable_monitoring,
                **kwargs,
            ),
            echo=echo,
        )
