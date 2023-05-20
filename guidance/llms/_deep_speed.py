import os
import time
import collections
import regex
import pygtrie
import queue
import threading
import logging
from ._transformers import Transformers, TransformersSession

class DeepSpeed(Transformers):
    """ A DeepSpeed accelerated language model with Guidance support.
    """

    cache = Transformers._open_cache("_deep_speed.diskcache")

    def __init__(self, model=None, tokenizer=None, caching=True, token_healing=True, acceleration=True, temperature=0.0, device=None, **deep_speed_kwargs):
        assert False, "DeepSpeed does not yet work! (this class is a work in progress and is currently waiting on supporting changes in DeepSpeed)"
        # parse alrady wrapped deepspeed objects
        import deepspeed
        if isinstance(model, deepspeed.InferenceEngine):
            self.wrapped_model_obj = model
            model = model.module
            assert len(deep_speed_kwargs) == 0, "You can't pass new deepspeed.init_inference kwargs if the passed model has already been initialized with DeepSpeed!"
        else:
            self.wrapped_model_obj = None

        # call the standard transformers init
        super().__init__(model=model, tokenizer=tokenizer, caching=caching, token_healing=token_healing, acceleration=acceleration, temperature=temperature, device=device)
        
        # wrap the underlying transformers model with DeepSpeed if still needed
        if self.wrapped_model_obj is None:
            self.wrapped_model_obj = deepspeed.init_inference(
                self.model_obj,
                **deep_speed_kwargs
            )
        self._generate_call = self.wrapped_model_obj.generate
        self.model_obj = self.wrapped_model_obj.module


class DeepSpeedSession(TransformersSession):
    pass