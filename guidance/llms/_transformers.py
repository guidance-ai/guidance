import os
import time
import time
import collections
import regex
import pygtrie
import queue
import threading
import logging
from ._llm import LLM, LLMSession

class Transformers(LLM):
    """ A HuggingFace transformers language model with Guidance support.
    """

    cache = LLM._open_cache("_transfomers.diskcache")

    def __init__(self, model=None, tokenizer=None, caching=True, token_healing=True, acceleration=True, temperature=0.0, device=None):
        super().__init__()

        # fill in default model value
        if model is None:
            model = os.environ.get("TRANSFORMERS_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.transformers_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        self.model_obj, self._tokenizer = self._model_and_tokenizer(model, tokenizer)

        self.model_name = model
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

    def encode(self, string, is_suffix=False, **kwargs):

        if is_suffix:
            string = self._tokenizer.bos_token + string

        if "return_tensors" in kwargs:
            out = self._tokenizer(string, **kwargs)
        else:
            out = self._tokenizer.encode(string, **kwargs)
        
        # remove the start token when we are encoding a suffix
        if is_suffix and out[0] == self._tokenizer.bos_token_id:
            if out[1] == self._tokenizer.bos_token_id:
                out = out[2:]
            else:
                out = out[1:]
        
        return out
    
    def decode(self, tokens, is_suffix=False, **kwargs):

        # Decode the string corresponding to a single suffix token.
        # Note that we need to decode after the start token for sentence-piece tokenizers so that white space is preserved
        if is_suffix:
            return self._tokenizer.decode([self._tokenizer.bos_token_id] + tokens)[len(self._tokenizer.bos_token):]
        else:
            return self._tokenizer.decode(tokens, **kwargs)

    def _build_token_prefix_map(self, model_name):
        """ Build a map from token to index.
        """
        token_map = pygtrie.CharTrie()
        for i in range(self._tokenizer.vocab_size):
            s = self.decode([i], is_suffix=True)
            if s in token_map:
                token_map[s].append(i) # handle duplicate token encodings... (GPT2 BPE has this oddly enough)
            else:
                token_map[s] = [i]

        return token_map

    def _model_and_tokenizer(self, model, tokenizer):

        # make sure transformers is installed
        try:
            import transformers
        except:
            raise Exception("Please install transformers with `pip install transformers` in order to use guidance.llms.Transformers!")

        # intantiate the model and tokenizer if needed
        if isinstance(model, str):
            if tokenizer is None:
                tokenizer = transformers.AutoTokenizer.from_pretrained(model)
            model = transformers.AutoModelForCausalLM.from_pretrained(model)
        
        assert tokenizer is not None, "You must give a tokenizer object when you provide a model object (as opposed to just a model name)!"
            
        return model, tokenizer

    def session(self):
        return TransformersSession(self)


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

                        # provide the past key values for the actual model call
                        model_kwargs["past_key_values"] = self._past_key_values
                        model_kwargs["position_ids"] = model_kwargs["position_ids"][:,len(self._prefix_cache):] # and update position ids

                        # we only need to do this first time, after that the past key values will
                        # be up until the last token, just like transformer models normally expect
                        # so we can clear our cache and let transformers cache like normal
                        self._prefix_cache = [] # this will get refilled once the generate call is done
                    
                        return model_kwargs
                    else:
                        return method(input_ids, **kwargs)
                decorate_prep_step.__func__ = method.__func__ # make us still look like a bound method
                return decorate_prep_step
            if getattr(self.llm.model_obj, "_orig_prepare_method", None) is None:
                self.llm.model_obj._orig_prepare_method = self.llm.model_obj.prepare_inputs_for_generation
            self.llm.model_obj.prepare_inputs_for_generation = prep_step_decorator(self.llm.model_obj._orig_prepare_method)

            # decorate the update step to save the past key values
            def update_step_decorator(method):
                def decorate_update_step(outputs, *args, **kwargs):

                    # save the past key values
                    self._past_key_values = outputs.past_key_values

                    return method(outputs, *args, **kwargs)
                return decorate_update_step
            if getattr(self.llm.model_obj, "_orig_update_method", None) is None:
                self.llm.model_obj._orig_update_method = self.llm.model_obj._update_model_kwargs_for_generation
            self.llm.model_obj._update_model_kwargs_for_generation = update_step_decorator(self.llm.model_obj._orig_update_method)

        return self

    def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None, top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=False, cache_seed=0):
        """ Generate a completion of the given prompt.
        """
        key = self.llm._cache_key(locals())
        
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

        # handle caching
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

            # save what the prompt looks like when coded and then decoded (this captures added start tokens, etc.)
            coded_prompt = self.llm.decode(input_ids[0])

            # setup token healing
            if token_healing:
                # pop off the last token since we will regen it
                last_token_id = input_ids[0][-1]
                last_token_str = self.llm.decode([last_token_id], is_suffix=True)
                healer = TokenHealingLogitsProcessor(self.llm, model_config.vocab_size, last_token_str)
                if healer.should_bias:
                    # tokenizer2.decode([tokenizer2.bos_token_id, 9288])[len(tokenizer2.bos_token):]
                    input_ids = input_ids[:,:-1]
                    attention_mask = attention_mask[:,:-1]
                    max_tokens += 1 # add one for the token we regen for token healing
                    processors.append(healer)
                else:
                    last_token_str = ""

            # make sure we don't run off the end of the model
            max_context = (getattr(model_config, "max_sequence_length", None) or getattr(model_config, "n_positions"))
            if max_tokens + len(input_ids[0]) > max_context:
                max_tokens = max_context - len(input_ids[0])

            # find how much of the prompt is cached
            for prefix_match_len, token in enumerate(input_ids[0]):
                if prefix_match_len >= len(self._prefix_cache) or token != self._prefix_cache[prefix_match_len]:
                    break

            # trim the cache to what we can use
            if prefix_match_len > 0 and prefix_match_len < len(self._prefix_cache):
                self._past_key_values = tuple((key[:,:,:prefix_match_len,:],value[:,:,:prefix_match_len,:]) for key,value in self._past_key_values) # TODO: this is specific to the GPT2 tensor layout
                self._prefix_cache = self._prefix_cache[:prefix_match_len]

            # see if we need to returns the scores
            output_scores = logprobs is not None and logprobs > 0

            # position_ids = torch.arange(prefix_match_len, input_ids.shape[-1], dtype=torch.long).unsqueeze(0)
                
            # trim input ids that we will pull from the cache instead of computing keys and values for
            # input_ids = input_ids[:,prefix_match_len:]

            # add support for pattern guidance
            if pattern is not None:
                processors.append(RegexLogitsProcessor(pattern, stop_regex, self.llm.decode, model_config.vocab_size, temperature == 0, len(coded_prompt), model_config.eos_token_id))

            if stop_regex is not None:
                stoppers.append(RegexStoppingCriteria(stop_regex, self.llm.decode, len(coded_prompt)))

            # a streamer to handle potentially partial output
            streamer = TransformersStreamer(
                input_ids=input_ids,
                stop_regex=stop_regex,
                last_token_str=last_token_str,
                coded_prompt=coded_prompt,
                llm=self.llm,
                max_new_tokens=max_tokens,
                lobprobs=logprobs
            )

            # the args for the transformers generate call
            generate_args = dict(
                inputs=input_ids,
                attention_mask=attention_mask,
                # position_ids=position_ids,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_p=top_p,
                pad_token_id=model_config.pad_token_id if model_config.pad_token_id is not None else model_config.eos_token_id,
                logits_processor=transformers.LogitsProcessorList(processors),
                stopping_criteria=transformers.StoppingCriteriaList(stoppers),
                # past_key_values=self._past_key_values,
                output_scores=logprobs is not None and logprobs > 0,
                return_dict_in_generate=True
            )

            # if we are streaming then we need to run the inference process in a separate thread
            if stream:
                generate_args["streamer"] = streamer
                thread = threading.Thread(target=self.llm.model_obj.generate, kwargs=generate_args)
                thread.start()
                return self._stream_then_save(streamer, key, thread)

            # if we are not streaming we still manually use the streamer for consistency
            else:
                generated_sequence = self.llm.model_obj.generate(**generate_args)
                streamer.put(generated_sequence)
                self.llm.cache[key] = streamer.__next__()
                self._update_prefix_cache(streamer)
        return self.llm.cache[key]
    
    def _update_prefix_cache(self, streamer):
        # note what we now have cached and ready for our next call in this session
        if self._past_key_values and len(streamer.generated_sequence) == 1:
            self._prefix_cache = streamer.generated_sequence[0][:self._past_key_values[0][0].shape[2]] # self._past_key_values is already saved, this just aligns with it

    def _stream_then_save(self, streamer, key, thread):
        list_out = []
        for out in streamer:
            list_out.append(out)
            yield out
        thread.join() # clean up the thread
        self.llm.cache[key] = list_out
        self._update_prefix_cache(streamer)

    def __exit__(self, exc_type, exc_value, traceback):
        """ Restore the model to its original state by removing monkey patches.
        """
        if getattr(self.llm.model_obj, "_orig_prepare_method", None) is not None:
            self.llm.model_obj.prepare_inputs_for_generation = self.llm.model_obj._orig_prepare_method
            del self.llm.model_obj._orig_prepare_method
        if getattr(self.llm.model_obj, "_orig_update_method", None) is not None:
            self.llm.model_obj._update_model_kwargs_for_generation = self.llm.model_obj._orig_update_method
            del self.llm.model_obj._orig_update_method
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

    def __init__(self, model, vocab_size, last_token_str, bias_value=50.):
        """ Build a new TokenHealingLogitsProcessor.

        Note that bias value is in score space (log-odds normally) and should be
        enough to ensure those tokens are the only ones used. But not so high
        as to destroy numerical precision.
        """
        import torch
        # last_token_str = model._tokenizer.decode([model._tokenizer.bos_token_id, last_token_id])[len(model._tokenizer.bos_token):]
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

class TransformersStreamer():
    def __init__(self, input_ids, stop_regex, last_token_str, coded_prompt, llm, max_new_tokens, lobprobs, timeout=None):
        self.timeout = timeout
        self.input_ids = input_ids
        self.stop_regex = stop_regex
        self.logprobs = lobprobs
        self.last_token_str = last_token_str
        # self.coded_prompt = coded_prompt
        self.llm = llm
        self.max_total_tokens = max_new_tokens + len(input_ids[0])
        coded_prompt = coded_prompt[:len(coded_prompt)-len(last_token_str)] # strip off the last token which will be regenerated
        self.str_pos = len(coded_prompt) + len(self.last_token_str)
        self.out_queue = queue.Queue()
        self.sequence_pos = [len(self.input_ids[0]) for i in range(len(self.input_ids))]
        self.generated_sequence = [[] for i in range(len(self.input_ids))]
        self.generated_scores = [[] for i in range(len(self.input_ids))]
        self.generated_string = [coded_prompt for i in range(len(self.input_ids))]
        self.prefix_cache = []

    def put(self, token_obj):

        import torch
        if isinstance(token_obj, torch.Tensor):
            new_tokens = token_obj
        else:
            new_tokens = token_obj['sequences']
        
        # if we are given a single sequence, then make it a batch of size 1
        if len(new_tokens.shape) == 1:
            new_tokens = new_tokens.unsqueeze(0)
        
        # extract the scores if we are given them (and format them to be the same shape as the tokens)
        if self.logprobs:
            assert len(new_tokens) == 1, "logprobs are not supported for batched generation right now in guidance.llms.Transformers"
            new_scores = list(token_obj['scores'])
            len_diff = len(new_tokens[0]) - len(new_scores)
            if len_diff > 0:
                new_scores = [None for i in range(len_diff)] + new_scores
            new_scores = [new_scores]
        
        out = {"choices": [None for i in range(len(self.input_ids))]}
        put_data = False
        for i in range(len(self.input_ids)):
            self.generated_sequence[i].extend(list(new_tokens[i]))
            if self.logprobs:
                self.generated_scores[i].extend(list(new_scores[i]))

            if self.sequence_pos[i] < len(self.generated_sequence[i]):
                display_tokens = list(self.generated_sequence[i][self.sequence_pos[i]:])
                val = self.llm.decode([self.llm._tokenizer.bos_token_id] + display_tokens)[len(self.llm._tokenizer.bos_token):]
                self.generated_string[i] += val
                
                if self.str_pos < len(self.generated_string[i]):
                    
                    display_logprobs = None
                    if self.logprobs:
                        display_scores = self.generated_scores[i][self.sequence_pos[i]:]
                        display_logprobs = []
                        for k in range(len(display_scores)):
                            top_inds = display_scores[k][0].argsort(descending=True)[:self.logprobs] # TODO: verify the [0] is always correct
                            display_logprobs.append({self.llm.decode([j], is_suffix=True): display_scores[k][0][j] for j in top_inds})

                    val = self.generated_string[i][self.str_pos:]
                    finish_reason = None
                    
                    if len(self.generated_sequence[i]) >= self.max_total_tokens:
                        finish_reason = "length"
                    elif self.generated_sequence[i][-1] == self.llm.model_obj.config.eos_token_id:
                        finish_reason = "endoftext"

                    # trim off the stop regex matches if needed
                    found_partial = False
                    stop_pos = len(val) + 1
                    if self.stop_regex is not None and finish_reason is None:
                        stop_regex_obj = [regex.compile(s) for s in self.stop_regex]
                        for s in stop_regex_obj:
                            m = s.search(val, partial=True)
                            if m:
                                span = m.span()
                                if span[1] > span[0]:
                                    if m.partial: # we might be starting a stop sequence, so we can't emit anything yet
                                        found_partial = True
                                        break
                                    else:
                                        stop_pos = min(span[0], stop_pos)

                    # record the reason we stopped (if we have stopped)
                    if stop_pos <= len(val):
                        finish_reason = "stop"
                    
                    if not found_partial:
                        out["choices"][i] = {
                            "text": val[:stop_pos],
                            "finish_reason": finish_reason,
                            "logprobs": {"token_healing_prefix": self.last_token_str, "top_logprobs": display_logprobs}
                        }
                        self.str_pos = len(self.generated_string[i])
                        put_data = True
                self.sequence_pos[i] = len(self.generated_sequence[i])
        if put_data:
            self.out_queue.put(out)

    def end(self):

        # make sure we have flushed all of the data
        for i in range(len(self.input_ids)):
            assert self.str_pos >= len(self.generated_string[i]), "Not all data was flushed, this means generation stopped for an unknown reason!"
        
        self.out_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.out_queue.get(timeout=self.timeout)
        if value is None:
            raise StopIteration()
        else:
            return value
