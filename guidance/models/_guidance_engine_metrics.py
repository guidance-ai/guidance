from pydantic import BaseModel, NonNegativeInt


class GuidanceEngineMetrics(BaseModel):
    generated_tokens: NonNegativeInt = 0
    forced_tokens: NonNegativeInt = 0
    model_input_tokens: NonNegativeInt = 0
