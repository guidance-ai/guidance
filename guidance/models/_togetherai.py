from ._model import Chat, Instruct
from ._openai import OpenAIChatEngine, OpenAI, OpenAIInstructEngine, OpenAICompletionEngine
from .transformers._transformers import TransformersTokenizer


class TogetherAI(OpenAI):
    def __init__(self, model, tokenizer=None, echo=True, api_key=None, max_streaming_tokens=1000, timeout=0.5, compute_log_probs=False, engine_class=None, **kwargs):
        '''
        Build a new TogetherAI model object that represents a model in a given state.
        '''

        tokenizer = TransformersTokenizer(model=model, tokenizer=tokenizer, ignore_bos_token=True)

        if engine_class is None:
            engine_map = {
                TogetherAI: OpenAICompletionEngine,
                TogetherAICompletion: OpenAICompletionEngine,
                TogetherAIInstruct: OpenAIInstructEngine,
                TogetherAIChat: OpenAIChatEngine
            }
            for k in engine_map:
                if issubclass(self.__class__, k):
                    engine_class = engine_map[k]
                    break

        super().__init__(
            model, tokenizer, echo, api_key, max_streaming_tokens, timeout, compute_log_probs, engine_class, **kwargs
        )

class TogetherAICompletion(TogetherAI):
    pass


class TogetherAIInstruct(TogetherAI, Instruct):
    pass


class TogetherAIChat(TogetherAI, Chat):
    pass