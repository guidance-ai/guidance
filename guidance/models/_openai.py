from typing import Optional
from ._base import Model
from ._openai_base import BaseOpenAIInterpreter, OpenAIAudioMixin, OpenAIImageMixin

class OpenAI(Model):
    def __init__(
        self,
        model: str,
        echo: bool = True,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Build a new OpenAI model object that represents a model in a given state.

        Parameters
        ----------
        model : str
            The name of the OpenAI model to use (e.g. gpt-4o-mini).
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        api_key : None or str
            The OpenAI API key to use for remote requests, passed directly to the `openai.OpenAI` constructor.

        **kwargs :
            All extra keyword arguments are passed directly to the `openai.OpenAI` constructor. Commonly used argument
            names include `base_url` and `organization`
        """

        if "audio-preview" in model:
            interpreter_cls = type("OpenAIAudioInterpreter", (BaseOpenAIInterpreter, OpenAIAudioMixin), {})
        elif model.startswith("gpt-4o") or model.startswith("o1"):
            interpreter_cls = type("OpenAIImageInterpreter", (BaseOpenAIInterpreter, OpenAIImageMixin), {})
        else:
            interpreter_cls = BaseOpenAIInterpreter

        super().__init__(
            interpreter=interpreter_cls(model, api_key=api_key, **kwargs), echo=echo
        )
