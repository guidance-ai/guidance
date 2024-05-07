from pydantic import BaseModel, NonNegativeInt


class GuidanceMetrics(BaseModel):
    prompt_tokens: NonNegativeInt = 0
    generated_tokens: NonNegativeInt = 0
