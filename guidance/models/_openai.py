from typing import Optional

from guidance._schema import SamplingParams


from ._base import Model
from ._openai_base import (
    BaseOpenAIInterpreter,
    OpenAIClientWrapper,
    OpenAIAudioMixin,
    OpenAIImageMixin,
    OpenAIJSONMixin,
    OpenAIRegexMixin,
    OpenAIRuleMixin,
)


class OpenAIInterpreter(OpenAIRuleMixin, OpenAIJSONMixin, OpenAIRegexMixin, BaseOpenAIInterpreter):
    def __init__(
        self,
        model: str,        
        default_sampling_params: Optional[SamplingParams],
        api_key: Optional[str] = None,
        **kwargs,
    ):
        try:
            import openai
        except ImportError:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )
                    
        client = openai.OpenAI(api_key=api_key, **kwargs)
        super().__init__(model=model, client=OpenAIClientWrapper(client), default_sampling_params=default_sampling_params)


class OpenAI(Model):
    def __init__(
        self,
        model: str,
        default_sampling_params: Optional[SamplingParams] = None,
        echo: bool = True,        
        *,
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
            interpreter_cls = type(
                "OpenAIAudioInterpreter", (OpenAIAudioMixin, OpenAIInterpreter), {}
            )
        elif model.startswith("gpt-4o") or model.startswith("o1"):
            interpreter_cls = type(
                "OpenAIImageInterpreter", (OpenAIImageMixin, OpenAIInterpreter), {}
            )
        else:
            interpreter_cls = OpenAIInterpreter

        super().__init__(interpreter=interpreter_cls(model, api_key=api_key, default_sampling_params=default_sampling_params, **kwargs), echo=echo)
