from ._model import Chat
from ._grammarless import GrammarlessEngine, Grammarless


class AzureAIStudioChatEngine(GrammarlessEngine):
    def __init__(
        self,
        *,
        tokenizer,
        max_streaming_tokens: int,
        timeout: float,
        compute_log_probs: bool,
        azureai_studio_endpoint: str,
        azureai_studio_key: str,
    ):
        self._endpoint = azureai_studio_endpoint
        self._api_key = azureai_studio_key

        super().__init__(tokenizer, max_streaming_tokens, timeout, compute_log_probs)


class AzureAIStudioChat(Grammarless, Chat):
    def __init__(
        self,
        azureai_studio_endpoint: str,
        azureai_studio_key: str,
        tokenizer=None,
        echo: bool = True,
        max_streaming_tokens: int = 1000,
        timeout: float = 0.5,
        compute_log_probs: bool = False,
    ):
        super().__init__(
            AzureAIStudioChatEngine(
                azureai_studio_endpoint=azureai_studio_endpoint,
                azureai_studio_key=azureai_studio_key,
                tokenizer=tokenizer,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
            ),
            echo=echo,
        )
