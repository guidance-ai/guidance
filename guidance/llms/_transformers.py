import openai
import pathlib
import diskcache
import os
import time
import requests
import warnings
import time
import collections
import tiktoken
import asyncio
import re
import regex
import pygtrie
import logging
from ._llm import LLM, LLMSession

class Transformers(LLM):
    """ A HuggingFace transformers language model with Guidance support.
    """

    cache = LLM._open_cache("_transfomers.diskcache")

    def __init__(self, model=None, caching=True, token_healing=True, acceleration=True, temperature=0.0, device=None):
        super().__init__()

        try:
            import transformers
        except:
            raise Exception("Please install transformers with `pip install transformers` in order to use guidance.llms.Transformers!")

        # fill in default model value
        if model is None:
            model = os.environ.get("TRANSFORMERS_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.transformers_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        self._encoding = transformers.AutoTokenizer.from_pretrained(model)
        
        self.model_name = model
        self.model_obj = transformers.AutoModelForCausalLM.from_pretrained(model)
        self.caching = caching
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.token_healing = token_healing
        self.acceleration = acceleration
        self.device = device
        if self.device is not None:
            self.model_obj = self.model_obj.to(self.device)

        self.token_prefix_map = self._build_token_prefix_map(model)

    def encode(self, string, **kwargs):
        if "return_tensors" in kwargs:
            return self._encoding(string, **kwargs)
        return self._encoding.encode(string, **kwargs)

    def _build_token_prefix_map(self, model_name):
        """ Build a map from token to index.
        """
        token_map = pygtrie.CharTrie()
        for i in range(self._encoding.vocab_size):
            s = self._encoding.decode([i])
            if s in token_map:
                token_map[s].append(i) # handle duplicate token encodings... (GPT2 BPE has this oddly enough)
            else:
                token_map[s] = [i]

        return token_map

    def session(self):
        return TransformersSession(self)
    
    def stream_then_save(self, gen, key):
        list_out = []
        for out in gen:
            list_out.append(out)
            yield out
        self.cache[key] = list_out

class TransformersSession(LLMSession):
    def __init__(self, llm):
        super().__init__(llm)
        
        self._past_key_values = None
        self._prefix_cache = []
    
    def __enter__(self):

        # we only need decorators if we are using token acceleration
        if self.llm.acceleration:

            # decorate the prep step to preserve the initial past key values we have passed
            def prep_step_decorator(method):
                def decorate_prep_step(input_ids, **kwargs):

                    # if we are extending the input ids with the cached tokens then
                    # don't pass past key values to the input prep step, otherwise it
                    # would delete all but the last input_ids, and we have already removed
                    # the correct prefix from the input_ids (which is not always all but the last one)
                    if len(self._prefix_cache) > 0:
                        
                        kwargs["past"] = None
                        input_ids = input_ids[:,len(self._prefix_cache):]
                        # if "attention_mask" in kwargs:
                        #     kwargs["attention_mask"] = kwargs["attention_mask"][:,len(self._prefix_cache):]
                        model_kwargs = method(input_ids, **kwargs)

                        # restore the past key values for the actual model call
                        model_kwargs["past_key_values"] = self._past_key_values

                        # we only need to do this first time, after that the past key values will
                        # be up until the last token, just let transformer models normally expect
                        # so we can clear our cache and let transformers cache like normal
                        self._prefix_cache = [] # this will get refilled once the generate call is done
                    
                        return model_kwargs
                    else:
                        return method(input_ids, **kwargs)
                return decorate_prep_step
            self._prev_prepare_method = self.llm.model_obj.prepare_inputs_for_generation
            self.llm.model_obj.prepare_inputs_for_generation = prep_step_decorator(self.llm.model_obj.prepare_inputs_for_generation)

            # decorate the update step to save the past key values
            def update_step_decorator(method):
                def decorate_update_step(outputs, *args, **kwargs):

                    # save the past key values
                    self._past_key_values = outputs.past_key_values

                    return method(outputs, *args, **kwargs)
                return decorate_update_step
            self._prev_update_method = self.llm.model_obj._update_model_kwargs_for_generation
            self.llm.model_obj._update_model_kwargs_for_generation = update_step_decorator(self.llm.model_obj._update_model_kwargs_for_generation)

        return self

    def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None, top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=False, cache_seed=0):
        """ Generate a completion of the given prompt.
        """

        # fill in defaults
        if temperature is None:
            temperature = self.llm.temperature
        if token_healing is None:
            token_healing = self.llm.token_healing

        if stop is not None:
            if isinstance(stop, str):
                stop_regex = [regex.escape(stop)]
            else:
                stop_regex = [regex.escape(s) for s in stop]
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]

        # assert healing, "Turning off token healing is not yet supported for the Transformers LLM"

        # handle caching
        key = "_---_".join([str(v) for v in (self.llm.model_name, prompt, stop_regex, temperature, n, max_tokens, logprobs, top_p, echo, logit_bias, token_healing, pattern, cache_seed)])
        if key not in self.llm.cache or not self.llm.caching:
            import transformers
            import torch
            # encode the prompt
            encoded = self.llm.encode(prompt, return_tensors="pt")
            if self.llm.device is not None:
                encoded = encoded.to(self.llm.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            model_config = self.llm.model_obj.config

            # ensure that we are extending a common sequence batch (our token healing assumes this right now)
            assert (input_ids[0,-1] == input_ids[:,-1]).all(), "The current token healing implementation assumes that batches are reps of the same sequence!"

            last_token_str = ""
            processors = []
            stoppers = []

            # setup token healing
            if token_healing:
                # pop off the last token since we will regen it
                last_token_id = input_ids[0][-1]
                healer = TokenHealingLogitsProcessor(self.llm, model_config.vocab_size, last_token_id)
                if healer.should_bias:
                    last_token_str = self.llm.decode([last_token_id])
                    input_ids = input_ids[:,:-1]
                    attention_mask = attention_mask[:,:-1]
                    max_tokens += 1 # add one for the token we regen for token healing
                    processors.append(healer)
                

            # make sure we don't run off the end of the model
            if max_tokens + len(input_ids[0]) > model_config.n_positions:
                max_tokens = model_config.n_positions - len(input_ids[0])

            # find how much of the prompt is cached
            for prefix_match_len, token in enumerate(input_ids[0]):
                if prefix_match_len >= len(self._prefix_cache) or token != self._prefix_cache[prefix_match_len]:
                    break

            # trim the cache to what we can use
            if prefix_match_len > 0 and prefix_match_len < len(self._prefix_cache):
                self._past_key_values = tuple((key[:,:,:prefix_match_len,:],value[:,:,:prefix_match_len,:]) for key,value in self._past_key_values) # TODO: this is specific to the GPT2 tensor layout
                self._prefix_cache = self._prefix_cache[:prefix_match_len]

            position_ids = torch.arange(prefix_match_len, input_ids.shape[-1], dtype=torch.long).unsqueeze(0)
                
            # trim input ids that we will pull from the cache instead of computing keys and values for
            # input_ids = input_ids[:,prefix_match_len:]

            # add support for pattern guidance
            if pattern is not None:
                processors.append(RegexLogitsProcessor(pattern, stop_regex, self.llm.decode, model_config.vocab_size, temperature == 0, len(prompt), model_config.eos_token_id))

            if stop_regex is not None:
                stoppers.append(RegexStoppingCriteria(stop_regex, self.llm.decode, len(prompt)))

            # call the model
            generated_sequence = self.llm.model_obj.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_p=top_p,
                pad_token_id=model_config.pad_token_id if model_config.pad_token_id is not None else model_config.eos_token_id,
                logits_processor=transformers.LogitsProcessorList(processors),
                stopping_criteria=transformers.StoppingCriteriaList(stoppers),
                past_key_values=self._past_key_values
            )

            # note what we now have cached and ready for our next call in this session
            if self._past_key_values:
                self._prefix_cache = generated_sequence[0][:self._past_key_values[0][0].shape[2]] # self._past_key_values is already saved, this just aligns with it

            # save the output. note we have to remove the input_ids prefix and the token healing prefix (last token str)
            out = {"choices": []}
            for i in range(len(input_ids)):
                generated_tokens = generated_sequence[i][len(input_ids[i]):]
                val = self.llm.decode(generated_tokens)[len(last_token_str):]
                
                # trim off the stop regex matches if needed
                stop_pos = len(val) + 1
                if stop_regex is not None:
                    stop_regex_obj = [regex.compile(s) for s in stop_regex]
                    for s in stop_regex_obj:
                        m = s.search(val)
                        if m:
                            stop_pos = min(m.span()[0], stop_pos)

                # record the reason we stopped
                if stop_pos <= len(val):
                    finish_reason = "stop"
                elif len(generated_tokens) >= max_tokens:
                    finish_reason = "length"
                elif generated_tokens[-1] == model_config.eos_token_id:
                    finish_reason = "endoftext"
                
                out["choices"].append({"text": val[:stop_pos], "finish_reason": finish_reason})

            if stream:
                return self.stream_then_save(out, key)
            else:
                self.llm.cache[key] = out
        return self.llm.cache[key]
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.llm.acceleration:
            self.llm.model_obj.prepare_inputs_for_generation = self._prev_prepare_method
            self.llm.model_obj._update_model_kwargs_for_generation = self._prev_update_method
        return False



class TokenHealingLogitsProcessor():
    """ Token healing.

    When we tokenize the prompt the last token we get is not the last token we would
    have gotten if the prompt + generation was concatented and then tokenized. This
    is not good because it does not align with the pretraining of the model, so
    we "heal" this boundary by backing up one token and then forcing the first token
    generated to start with the prefix of the last token in the prompt. This could
    result in the same token as the end of the prompt, or another longer one.
    """

    def __init__(self, model, vocab_size, last_token_id, bias_value=50.):
        """ Build a new TokenHealingLogitsProcessor.

        Note that bias value is in score space (log-odds normally) and should be
        enough to ensure those tokens are the only ones used. But not so high
        as to destroy numerical precision.
        """
        import torch
        last_token_str = model._encoding.decode(last_token_id)
        allowed_first_tokens = [v for arr in model.token_prefix_map.values(prefix=last_token_str) for v in arr]
        assert len(allowed_first_tokens) > 0, "Error in token healing map! No match found for: `"+last_token_str+"`"
        
        # if we have multiple possible completions past the last token, then biasing is needed
        if len(allowed_first_tokens) > 1:
            self.first_token_mask = torch.zeros(vocab_size)
            self.first_token_mask.scatter_(0, torch.tensor(allowed_first_tokens), bias_value)
            if model.device is not None:
                self.first_token_mask = self.first_token_mask.to(model.device)
            self.should_bias = True
        
        # otherwise we have nothing to do (the last token is already unique)
        else:
            self.should_bias = False

    def __call__(self, input_ids, scores):

        # we only bias the first token generated
        if not self.should_bias:
            return scores
        self.should_bias = False
        
        # make only allowed tokens possible
        return scores + self.first_token_mask
    
class RegexLogitsProcessor():
    """ Pattern guiding.
    
    Guide generation to match a regular expression.
    TODO: currently slow, could be made much faster by doing rejection sampling inline with the sampling/greedy process.
    """

    def __init__(self, pattern, stop_regex, decode, vocab_size, is_greedy, prefix_length, eos_token_id, max_consider=10000, bias_value=30.):
        """ Build a new TokenHealingLogitsProcessor.

        Parameters
        ----------
        pattern : str
            The regex pattern we are seeking to match.
        stop_regex : str
            The stop regex that is allowed to come after this pattern.
        decode : function
            The token decoding function.
        vocab_size : int
            The size of the vocabulary.
        is_greedy : bool
            The token selection mode currently in use. We need to know this so we can
            effectively take over that sampling process inside this logit processor.
        max_consider : int
            How many top values to bias. Note that we could remove this option once this
            processor is performance optimized (by integrating it into the sampling/greedy process).
        bias_value : float
            The bias value is in score space (log-odds normally) and should be
            enough to ensure those tokens are the only ones used. But not so high
            as to destroy numerical precision.
        eos_token_id : int
            The end of the stop token of the model.
        """
        import torch
        
        if stop_regex is not None:
            pattern += "(" + "|".join(stop_regex) + ")?"
        stop_token_str = regex.escape(decode([eos_token_id]))
        pattern += "(" + stop_token_str + ")?"
        self.pattern = regex.compile(pattern)
        self.decode = decode
        self.is_greedy = is_greedy
        self.prefix_length = prefix_length
        self.max_consider = max_consider
        self.bias_value = bias_value
        self.bias_vector = torch.zeros(vocab_size)
        self.current_strings = None
        self.current_length = 0
        self.forced_chars = 0

    def __call__(self, input_ids, scores):
        import torch

        # extend our current strings
        if self.current_strings is None:
            self.current_strings = ["" for _ in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i] += self.decode(input_ids[i][self.current_length:])

        # trim off the prefix string so we don't look for stop matches in the prompt
        if self.current_length == 0:
            self.forced_chars = self.prefix_length - len(self.current_strings[i]) # account for token healing forced prefixes
            for i in range(len(self.current_strings)):
                self.current_strings[i] = self.current_strings[i][self.prefix_length:]

        self.current_length = len(input_ids[0])
        
        # compute the bias values
        self.bias_vector[:] = 0
        sort_inds = torch.argsort(scores, 1, True)
        for i in range(min(sort_inds.shape[1], self.max_consider)):
            m = self.pattern.fullmatch((self.current_strings[0] + self.decode([sort_inds[0,i]]))[self.forced_chars:], partial=True) # partial means we don't match currently but might as the string grows
            if m:
                self.bias_vector[sort_inds[0,i]] = self.bias_value
                if self.is_greedy:
                    break # we are done if we are doing greedy sampling and we found the top valid hit
        
        # make only allowed tokens
        return scores + self.bias_vector

class RegexStoppingCriteria():
    def __init__(self, stop_pattern, decode, prefix_length):
        if isinstance(stop_pattern, str):
            self.stop_patterns = [regex.compile(stop_pattern)]
        else:
            self.stop_patterns = [regex.compile(pattern) for pattern in stop_pattern]
        self.prefix_length = prefix_length
        self.decode = decode
        self.current_strings = None
        self.current_length = 0

    def __call__(self, input_ids, scores, **kwargs):

        # extend our current strings
        if self.current_strings is None:
            self.current_strings = ["" for _ in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i] += self.decode(input_ids[i][self.current_length:])
        
        # trim off the prefix string so we don't look for stop matches in the prompt
        if self.current_length == 0:
            for i in range(len(self.current_strings)):
                self.current_strings[i] = self.current_strings[i][self.prefix_length:]
        
        self.current_length = len(input_ids[0])
        
        # check if all of the strings match a stop string (and hence we can stop the batch inference)
        all_done = True
        for i in range(len(self.current_strings)):
            found = False
            for s in self.stop_patterns:
                if s.search(self.current_strings[i]):
                    found = True
            if not found:
                all_done = False
                break
        
        return all_done


