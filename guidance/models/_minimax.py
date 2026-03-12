import os

from guidance._schema import SamplingParams

from ._base import Model
from ._openai_base import (
    BaseOpenAIInterpreter,
    OpenAIClientWrapper,
    OpenAIJSONMixin,
    OpenAIRegexMixin,
    OpenAIRuleMixin,
)

MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxInterpreter(OpenAIRuleMixin, OpenAIJSONMixin, OpenAIRegexMixin, BaseOpenAIInterpreter):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        try:
            import openai
        except ImportError as ie:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.MiniMax!"
            ) from ie

        if api_key is None:
            api_key = os.environ.get("MINIMAX_API_KEY")
        if base_url is None:
            base_url = os.environ.get("MINIMAX_API_BASE", MINIMAX_DEFAULT_BASE_URL)

        client = openai.OpenAI(api_key=api_key, base_url=base_url, **kwargs)
        super().__init__(model=model, client=OpenAIClientWrapper(client))


class MiniMax(Model):
    def __init__(
        self,
        model: str = "MiniMax-M2.5",
        sampling_params: SamplingParams | None = None,
        echo: bool = True,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        """Build a new MiniMax model object that represents a model in a given state.

        MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1.

        Parameters
        ----------
        model : str
            The name of the MiniMax model to use. Available models:
            - ``MiniMax-M2.5`` (default): Peak Performance, 204K context window.
            - ``MiniMax-M2.5-highspeed``: Same performance, faster and more agile, 204K context window.
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        api_key : None or str
            The MiniMax API key. If not provided, falls back to the ``MINIMAX_API_KEY`` environment variable.
        base_url : None or str
            The MiniMax API base URL. Defaults to ``https://api.minimax.io/v1``.
            The China endpoint is ``https://api.minimaxi.com/v1``.

        **kwargs :
            All extra keyword arguments are passed directly to the ``openai.OpenAI`` constructor.
        """
        super().__init__(
            interpreter=MiniMaxInterpreter(model, api_key=api_key, base_url=base_url, **kwargs),
            sampling_params=SamplingParams() if sampling_params is None else sampling_params,
            echo=echo,
        )
