from pydantic import BaseModel, NonNegativeInt


class GuidanceEngineMetrics(BaseModel):
    model_input_tokens: NonNegativeInt = 0
    model_output_tokens: NonNegativeInt = 0
