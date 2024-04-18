import tiktoken

from ._vertexai import VertexAICompletion, VertexAIInstruct, VertexAIChat

try:
    from vertexai.language_models import CodeGenerationModel, CodeChatModel

    is_vertexai = True
except ModuleNotFoundError:
    is_vertexai = False


class CodeyCompletion(VertexAICompletion):
    def __init__(
        self, model, tokenizer=None, echo=True, max_streaming_tokens=None, **kwargs
    ):

        if isinstance(model, str):
            model = CodeGenerationModel.from_pretrained(model)

        # Codey does not have a public tokenizer, so we pretend it tokenizes like gpt2...
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")

        # the superclass does all the work
        super().__init__(
            model,
            tokenizer=tokenizer,
            echo=echo,
            max_streaming_tokens=max_streaming_tokens,
            **kwargs,
        )


class CodeyInstruct(VertexAIInstruct):
    def __init__(
        self, model, tokenizer=None, echo=True, max_streaming_tokens=None, **kwargs
    ):

        if isinstance(model, str):
            model = CodeGenerationModel.from_pretrained(model)

        # Codey does not have a public tokenizer, so we pretend it tokenizes like gpt2...
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")

        # the superclass does all the work
        super().__init__(
            model,
            tokenizer=tokenizer,
            echo=echo,
            max_streaming_tokens=max_streaming_tokens,
            **kwargs,
        )


class CodeyChat(VertexAIChat):
    def __init__(
        self, model, tokenizer=None, echo=True, max_streaming_tokens=None, **kwargs
    ):

        if isinstance(model, str):
            model = CodeChatModel.from_pretrained(model)

        # PaLM2 does not have a public tokenizer, so we pretend it tokenizes like gpt2...
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("gpt2")

        # the superclass does all the work
        super().__init__(
            model,
            tokenizer=tokenizer,
            echo=echo,
            max_streaming_tokens=max_streaming_tokens,
            **kwargs,
        )
