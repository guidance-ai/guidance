from pydantic import BaseModel, NonNegativeInt


class GuidanceEngineMetrics(BaseModel):
    prompt_tokens: NonNegativeInt = 0
    generated_tokens: NonNegativeInt = 0
    forced_tokens: NonNegativeInt = 0
