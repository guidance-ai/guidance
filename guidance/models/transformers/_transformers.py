import re
import functools
import time
import collections
import os

try:
    import torch
except ImportError:
    pass

from .._model import Chat
from .._local import Local


class Transformers(Local):
    def __init__(self, model=None, tokenizer=None, echo=True, caching=True, temperature=0.0, device=None, **kwargs):
        
        # fill in default model value
        if model is None:
            model = os.environ.get("TRANSFORMERS_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.transformers_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        self.model_obj, self._orig_tokenizer = self._model_and_tokenizer(model, tokenizer, **kwargs)

        if not isinstance(model, str):
            self.model = model.__class__.__name__
        self.caching = caching
        # self.current_time = time.time()
        # self.call_history = collections.deque()
        self.temperature = temperature
        if device is not None: # set the device if requested
            self.model_obj = self.model_obj.to(device)
        self.device = self.model_obj.device # otherwise note the current device

        # build out the set of byte_string tokens
        tkz = self._orig_tokenizer
        if hasattr(tkz, "byte_decoder"):
            byte_tokens = []
            for i in range(len(tkz)):
                byte_coded = bytes([tkz.byte_decoder[c] for c in tkz.convert_ids_to_tokens(i)])
                byte_tokens.append(byte_coded)
        else:
            byte_tokens = [bytes(tkz.convert_tokens_to_string(['a', tkz.convert_ids_to_tokens(i)])[1:], encoding="utf8") for i in range(len(tkz))]

        # the superclass does most of the work once we have the tokens
        super().__init__(
            byte_tokens,
            tkz.bos_token_id,
            tkz.eos_token_id,
            echo=echo
        )

        self._cache_state["past_key_values"] = None
        self._cache_state["logits"] = None
        self._cache_state["cache_token_ids"] = []

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):

        # intantiate the model and tokenizer if needed
        if isinstance(model, str):

            # make sure transformers is installed
            try:
                import transformers
            except:
                raise Exception("Please install transformers with `pip install transformers` in order to use guidance.models.Transformers!")

            if tokenizer is None:
                tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, **kwargs)
            model = transformers.AutoModelForCausalLM.from_pretrained(model, **kwargs)
        
        assert tokenizer is not None, "You must give a tokenizer object when you provide a model object (as opposed to just a model name)!"
            
        return model, tokenizer

    def _get_logits(self, token_ids, forced_bytes):
        '''Computes the logits for the given token state.
        
        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        '''

        # make sure we don't run off the end of the model
        if len(token_ids) >= getattr(self.model_obj.config, "max_position_embeddings", 1e10):
            raise Exception(f"Attempted to run a transformers model past its maximum context window size of {self.model_obj.config.max_position_embeddings}!")

        # get the number of cache positions we are using
        cache_token_ids = self._cache_state["cache_token_ids"]
        num_cached = 0
        for id in cache_token_ids:
            if num_cached >= len(cache_token_ids) or num_cached >= len(token_ids) or token_ids[num_cached] != id:
                break
            num_cached += 1

        # reset the cache length according to that number of positions
        past_key_values = self._cache_state["past_key_values"]
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        if past_length > num_cached:
            past_length = max(0, num_cached - 1) # note we recompute the last token because we don't bother to handle the special case of just computing logits
            self._cache_state["past_key_values"] = tuple(tuple(p[..., :past_length, :] for p in v) for v in past_key_values)
        cache_token_ids[past_length:] = []
        
        # call the model
        new_token_ids = token_ids[past_length:]
        if len(new_token_ids) > 0:
            with torch.no_grad():
                model_out = self.model_obj(
                    input_ids=torch.tensor(new_token_ids).unsqueeze(0).to(self.device),
                    past_key_values=self._cache_state["past_key_values"],
                    use_cache=True,
                    position_ids=torch.arange(past_length, past_length+len(new_token_ids)).unsqueeze(0).to(self.device),
                    attention_mask=torch.ones(1, past_length + len(new_token_ids)).to(self.device),
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False
                )

            # save the results
            self._cache_state["past_key_values"] = model_out.past_key_values
            cache_token_ids.extend(new_token_ids)
            self._cache_state["logits"] = model_out.logits[0, -1, :].cpu().numpy()
        
        return self._cache_state["logits"]
    
class TransformersChat(Transformers, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)