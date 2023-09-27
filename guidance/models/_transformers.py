import re
import functools
import time
import collections
import os

try:
    import torch
except ImportError:
    pass

from ._model import Chat
from ._local import Local


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

        # build the token set and pass that to the Local constructor
        # note that we prefix the tokens with a letter so some sentence peice tokenizers don't strip leading spaces.
        tkz = self._orig_tokenizer
        super().__init__(
            [tkz.convert_tokens_to_string(['a', tkz.convert_ids_to_tokens(i)])[1:] for i in range(tkz.vocab_size)],
            tkz.bos_token_id,
            tkz.eos_token_id,
            echo=echo
        )

        self._cache_state["past_key_values"] = None
        self._cache_state["logits"] = None

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):

        # intantiate the model and tokenizer if needed
        if isinstance(model, str):

            # make sure transformers is installed
            try:
                import transformers
            except:
                raise Exception("Please install transformers with `pip install transformers` in order to use guidance.llms.Transformers!")

            if tokenizer is None:
                tokenizer = transformers.AutoTokenizer.from_pretrained(model, **kwargs)
            model = transformers.AutoModelForCausalLM.from_pretrained(model, **kwargs)
        
        assert tokenizer is not None, "You must give a tokenizer object when you provide a model object (as opposed to just a model name)!"

        # discover how the model handles leading spaces
        # tokens = tokenizer.encode("alpha ruby")
        # raw_coded = ''.join([tokenizer.convert_ids_to_tokens(id) for id in tokens])
        # recoded = tokenizer.decode(tokens)
        # assert len(raw_coded) == len(recoded), "The tokenizer is changing the length of the string, so you need make a special subclass to handle this model!"
        # self.leading_space_token = raw_coded[-5]
            
        return model, tokenizer

    def _get_logits(self):
        '''Computes the logits for the given token state.
        
        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        '''

        cache_token_ids = self._cache_state["cache_token_ids"]
        new_token_ids = self._cache_state["new_token_ids"]

        # get the number of cache positions we are using
        past_key_values = self._cache_state["past_key_values"]
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        if past_length > len(cache_token_ids):
            past_length = len(cache_token_ids)-1
            self._cache_state["past_key_values"] = tuple(tuple(p[..., :past_length, :] for p in v) for v in past_key_values)

            # note we recompute the last token because we don't bother to handle the special case of just computing logits
            new_token_ids.insert(0, cache_token_ids[-1])
            cache_token_ids.pop()

        # call the model
        if len(new_token_ids) > 0:
            model_out = self.model_obj(
                input_ids=torch.tensor(new_token_ids).unsqueeze(0).to(self.device),
                past_key_values=self._cache_state["past_key_values"],
                use_cache=True,
                position_ids=torch.arange(past_length, past_length+len(new_token_ids)).unsqueeze(0).to(self.device),
                attention_mask=torch.ones(1, past_length + len(new_token_ids)).to(self.device)
            )

            # save the results
            self._cache_state["past_key_values"] = model_out.past_key_values
            cache_token_ids.extend(new_token_ids)
            self._cache_state["new_token_ids"].clear()
            self._cache_state["logits"] = model_out.logits[0, -1, :]
        
        return self._cache_state["logits"]
    
    # def __init__(self, engine, caching=True, **engine_kwargs):
    #     super().__init__(engine, caching=caching)
    #     # self.engine = engine

    #     if isinstance(self.engine, str):
    #         self.engine = guidance.endpoints.Transformers(engine, **engine_kwargs)
    #     # self._endpoint_session = self.endpoint.session()

class TransformersChat(Transformers, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)