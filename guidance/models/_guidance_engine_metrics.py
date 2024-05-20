from pydantic import BaseModel, NonNegativeInt


class GuidanceEngineMetrics(BaseModel):
    engine_input_tokens: NonNegativeInt = 0
    engine_output_tokens: NonNegativeInt = 0
