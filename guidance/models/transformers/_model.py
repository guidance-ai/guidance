from typing import Any, Optional

from ..base import ModelWithEngine
from ._engine import TransformersEngine
from ._state import TransformersChatState, TransformersMessage

# TODO: Expose a non-chat version
class Transformers(ModelWithEngine[TransformersChatState]):
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

    def initial_state(self) -> TransformersChatState:
        return TransformersChatState.from_model_id(self.engine.model)

    def build_prompt(self, state: TransformersChatState) -> str:
        state_dict = state.get_state()
        prefill = state_dict["prefill"]
        if prefill is None:
            role = state_dict["active_role"]
            if role is None:
                raise ValueError("Can't generate with no active role")
            prefill = {"role": "user", "content": ""}
        prompt = apply_chat_template(
            list(state_dict["messages"]),
            state_dict["prefill"],
            None,
            None,
            self.engine.tokenizer._orig_tokenizer,  # type: ignore[attr-defined]
        )
        return prompt


def apply_chat_template(
    messages: list[TransformersMessage],
    prefill: Optional[TransformersMessage],
    tools: Optional[list[Any]],
    chat_template: Optional[str],
    tokenizer: Any,
) -> str:
    if prefill is None:
        sentinel_value = None
    elif prefill["content"]:
        sentinel_value = None
        messages = messages + [prefill]
    else:
        # This is a hack to get around the fact that apply_chat_template won't properly continue the final message
        # if it is empty. We add a sentinel value to the final message, and then remove it after the fact.
        sentinel_value = "<|FINAL_MESSAGE_SENTINEL_VALUE|>"
        messages = messages + [dict(role=prefill["role"], content=sentinel_value)]
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=chat_template,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    if sentinel_value is not None:
        prompt = prompt[: prompt.rindex(sentinel_value)]
    return prompt
