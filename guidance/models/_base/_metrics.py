from pydantic import BaseModel, computed_field, Field

class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    """Number of cached tokens present in the prompt."""

class CompletionTokensDetails(BaseModel):
    fast_forward_tokens: int = 0
    """Number of fast-forward tokens used in the completion."""

class TokenUsage(BaseModel):
    completion_tokens: int = 0
    """Number of tokens in the completion."""

    prompt_tokens: int = 0
    """Number of tokens in the prompt."""

    completion_tokens_details: CompletionTokensDetails = Field(
        default_factory=CompletionTokensDetails,
    )
    """Details about the completion tokens, such as fast-forward tokens."""

    prompt_tokens_details: PromptTokensDetails = Field(
        default_factory=PromptTokensDetails,
    )
    """Details about the prompt tokens, such as cached tokens."""

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Total number of tokens used in the request (prompt + completion)."""
        return self.completion_tokens + self.prompt_tokens

