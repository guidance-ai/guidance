from typing import Optional, TypedDict


class SamplingParams(TypedDict):
    top_p: Optional[float]
    top_k: Optional[int]
    min_p: Optional[float]
    repetition_penalty: Optional[float]
