import os
import time
import collections
import regex
import pygtrie
import queue
import threading
import logging
from ._llm import LLM, LLMSession, SyncSession

class Transformers(LLM):
    """ A HuggingFace transformers language model with Guidance support.
    """

    cache = LLM._open_cache("_transformers.diskcache")

    def __init__(self, model=None, tokenizer=None, caching=True, token_healing=True, acceleration=True, \
                 temperature=0.0, device=None, **kwargs):
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

        self.model_obj, self._tokenizer = self._model_and_tokenizer(model, tokenizer, **kwargs)
        self._generate_call = self.model_obj.generate

        self.model_name = model
        self.caching = caching
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.token_healing = token_healing
        self.acceleration = acceleration
        if device is not None: # set the device if requested
            self.model_obj = self.model_obj.to(device)
        self.device = self.model_obj.device # otherwise note the current device

        self._prefix_ids = [self._tokenizer.bos_token_id, 100] # token ids that we use to decode tokens after a prefix
        self._prefix_str = self._tokenizer.decode(self._prefix_ids, fragment=False)

        self._token_prefix_map = self._build_token_prefix_map(model)

    def prefix_matches(self, prefix):
        """ Return the list of tokens that match the given prefix.
        """
        return [v for arr in self._token_prefix_map.values(prefix=prefix) for v in arr]

    def encode(self, string, fragment=True, **kwargs):

        if fragment:
            string = self._prefix_str + string

        if "return_tensors" in kwargs:
            out = self._tokenizer(string, **kwargs)
        else:
            out = self._tokenizer.encode(string, **kwargs)
        
        # remove the start token when we are encoding a suffix
        if fragment:
            if out[1] == self._tokenizer.bos_token_id: # sometime the tokenizer adds an extra start token
                out = out[3:]
            else:
                out = out[2:]
        
        return out
    
    def id_to_token(self, id):
        return self._tokenizer.convert_ids_to_tokens([id])[0]
    
    def token_to_id(self, token):
        return self._tokenizer.convert_tokens_to_ids([token])[0]
    
    # def role_start(self, role):
    #     """ The starting role tag for chat models.

    #     #TODO Right now this just assumes the StableLM syntax, but this should be expanded later.
    #     """
    #     return "<|"+role.upper()+"|>"
    
    # def role_end(self, role=None):
    #     return ""

    def end_of_text(self):
        return self._tokenizer.eos_token

    @staticmethod
    def role_start(role):
        raise NotImplementedError("In order to use chat role tags you need to use a chat-specific subclass of Transformers for your LLM from guidance.transformers.*!")
    
    def decode(self, tokens, fragment=True, **kwargs):

        # if the last token is the end of string token, or the first is a start of string we remove it because it cause odd spacing decoding of fragments
        add_eos = ""
        add_bos = ""
        if fragment:
            if len(tokens) > 0 and tokens[-1] == self._tokenizer.eos_token_id:
                add_eos = self._tokenizer.eos_token
                tokens = tokens[:-1]
            if len(tokens) > 0 and tokens[0] == self._tokenizer.bos_token_id:
                add_bos = self._tokenizer.bos_token
                tokens = tokens[1:]
        
        # Decode the string corresponding to a single suffix token.
        # Note that we need to decode after the start token for sentence-piece tokenizers so that white space is preserved
        if fragment:
            return add_bos + self._tokenizer.decode(self._prefix_ids + list(tokens))[len(self._prefix_str):] + add_eos
        else:
            return add_bos + self._tokenizer.decode(tokens, **kwargs) + add_eos

    def _build_token_prefix_map(self, model_name):
        """ Build a map from token to index.
        """
        token_map = pygtrie.CharTrie()
        for i in range(self._tokenizer.vocab_size):
            s = self.decode([i])
            if s in token_map:
                token_map[s].append(i) # handle duplicate token encodings... (GPT2 BPE has this oddly enough)
            else:
                token_map[s] = [i]

        return token_map

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):

        # make sure transformers is installed
        try:
            import transformers
        except:
            raise Exception("Please install transformers with `pip install transformers` in order to use guidance.llms.Transformers!")

        # intantiate the model and tokenizer if needed
        if isinstance(model, str):
            if tokenizer is None:
                tokenizer = transformers.AutoTokenizer.from_pretrained(model, **kwargs)
            model = transformers.AutoModelForCausalLM.from_pretrained(model, **kwargs)
        
        assert tokenizer is not None, "You must give a tokenizer object when you provide a model object (as opposed to just a model name)!"
            
        return model, tokenizer

    def session(self, asynchronous=False):
        if asynchronous:
            return TransformersSession(self)
        else:
            return SyncSession(TransformersSession(self))


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
                        if "position_ids" in model_kwargs: # models like OPT update the position ids internally
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
                    self._past_key_values = getattr(outputs, "past_key_values", None)

                    return method(outputs, *args, **kwargs)
                return decorate_update_step
            if getattr(self.llm.model_obj, "_orig_update_method", None) is None:
                self.llm.model_obj._orig_update_method = self.llm.model_obj._update_model_kwargs_for_generation
            self.llm.model_obj._update_model_kwargs_for_generation = update_step_decorator(self.llm.model_obj._orig_update_method)

        return self

    # def __call__(self, *args, **kwargs):
    #     return self.__call__(*args, **kwargs)
    
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None, top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=False, cache_seed=0, caching=None):
        """ Generate a completion of the given prompt.
        """
        
        # fill in defaults
        if temperature is None:
            temperature = self.llm.temperature
        if token_healing is None:
            token_healing = self.llm.token_healing

        # generate the cache key
        key = self._cache_key(locals())

        # set the stop patterns
        if stop is not None:
            if isinstance(stop, str):
                stop_regex = [regex.escape(stop)]
            else:
                stop_regex = [regex.escape(s) for s in stop]
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]
        if stop_regex is None:
            stop_regex = []
        stop_regex.append(regex.escape(self.llm._tokenizer.eos_token)) # make sure the end of sequence token is always included

        # handle caching
        in_cache = key in self.llm.cache
        not_caching = (caching is not True and not self.llm.caching) or caching is False
        if not in_cache or not_caching:
            import transformers

            assert prompt != "", "You must provide a non-zero length prompt to the Transformers language model!"

            # encode the prompt
            encoded = self.llm.encode([prompt for _ in range(n)], return_tensors="pt", fragment=False)
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
                last_token_str = self.llm.decode([last_token_id])
                healer = TokenHealingLogitsProcessor(self.llm, model_config.vocab_size, last_token_str)
                if healer.should_bias:
                    input_ids = input_ids[:,:-1]
                    attention_mask = attention_mask[:,:-1]
                    max_tokens += 1 # add one for the token we regen for token healing
                    processors.append(healer)
                else:
                    last_token_str = ""

            # setup logit biasing
            if logit_bias is not None:
                processors.append(BiasLogitsProcessor(self.llm, model_config.vocab_size, logit_bias))

            # make sure we don't run off the end of the model
            max_context = (getattr(model_config, "max_sequence_length", None) or getattr(model_config, "max_seq_len", None) or getattr(model_config, "n_positions", None) or getattr(model_config, "max_position_embeddings"))
            if max_tokens + len(input_ids[0]) > max_context:
                max_tokens = max_context - len(input_ids[0])

            # find how much of the prompt is cached
            prefix_match_len = 0
            for token in input_ids[0]:
                if prefix_match_len >= len(self._prefix_cache) or token != self._prefix_cache[prefix_match_len]:
                    break
                else:
                    prefix_match_len += 1

            # we always need to run the model on at least one token so transformers is happy
            if prefix_match_len == len(input_ids[0]):
                prefix_match_len -= 1

            # trim the cache to what we can use
            if prefix_match_len < len(self._prefix_cache): # prefix_match_len > 0 and 
                self._past_key_values = tuple((key[:,:,:prefix_match_len,:],value[:,:,:prefix_match_len,:]) for key,value in self._past_key_values) # TODO: this is specific to the GPT2 tensor layout
                self._prefix_cache = self._prefix_cache[:prefix_match_len]

            # see if we need to returns the scores
            # output_scores = logprobs is not None and logprobs > 0

            # position_ids = torch.arange(prefix_match_len, input_ids.shape[-1], dtype=torch.long).unsqueeze(0)
                
            # trim input ids that we will pull from the cache instead of computing keys and values for
            # input_ids = input_ids[:,prefix_match_len:]

            # add support for pattern guidance
            if pattern is not None:
                processors.append(RegexLogitsProcessor(pattern, stop_regex, self.llm.decode, model_config.vocab_size, temperature == 0, len(coded_prompt), self.llm._tokenizer.eos_token_id))

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
                pad_token_id=model_config.pad_token_id if model_config.pad_token_id is not None else self.llm._tokenizer.eos_token_id,
                logits_processor=transformers.LogitsProcessorList(processors),
                stopping_criteria=transformers.StoppingCriteriaList(stoppers),
                # past_key_values=self._past_key_values,
                output_scores=logprobs is not None and logprobs > 0,
                return_dict_in_generate=True
            )

            # override the model config for do_sample when the temperature requires it
            do_sample = getattr(self.llm.model_obj.config, "do_sample", None)
            if do_sample is True and temperature == 0:
                generate_args["do_sample"] = False
            elif do_sample is False and temperature > 0:
                generate_args["do_sample"] = True

            # if we are streaming then we need to run the inference process in a separate thread
            if stream:
                generate_args["streamer"] = streamer
                thread = threading.Thread(target=self.llm._generate_call, kwargs=generate_args)
                thread.start()
                return self._stream_then_save(streamer, key, thread)

            # if we are not streaming we still manually use the streamer for consistency
            else:
                generated_sequence = self.llm._generate_call(**generate_args)
                streamer.put(generated_sequence)
                self.llm.cache[key] = streamer.__next__()
                self._update_prefix_cache(streamer)
        return self.llm.cache[key]
    
    def _update_prefix_cache(self, streamer):
        # note what we now have cached and ready for our next call in this session
        if self._past_key_values and len(streamer.generated_sequence) == 1:
            self._prefix_cache = streamer.generated_sequence[0][:self._past_key_values[0][0].shape[-2]] # self._past_key_values is already saved, this just aligns with it

    def _stream_then_save(self, streamer, key, thread):
        list_out = []
        for out in streamer:
            list_out.append(out)
            yield out
        thread.join() # clean up the thread
        self.llm.cache[key] = list_out
        self._update_prefix_cache(streamer)
        self._last_computed_key = key

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

        try:
            allowed_first_tokens = model.prefix_matches(last_token_str)
            assert len(allowed_first_tokens) > 0, "Error in token healing map! No match found for: `"+last_token_str+"`"
        except KeyError:
            # this must be a special token outside the vocab, so we assume it does not have any valid extensions
            allowed_first_tokens = []
        
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
    
class BiasLogitsProcessor():
    """ Simple token biasing.
    """

    def __init__(self, model, vocab_size, logit_bias):
        """ Build a new BiasLogitsProcessor.
        """
        import torch
        
        self.bias_vector = torch.zeros(vocab_size)
        for token, bias in logit_bias.items():
            self.bias_vector[token] = bias
        self.bias_vector = self.bias_vector.to(model.device)

    def __call__(self, input_ids, scores):
        return scores + self.bias_vector
    
class RegexLogitsProcessor():
    """ Pattern guiding.
    
    Guide generation to match a regular expression.
    TODO: currently slow, could be made much faster by doing rejection sampling inline with the sampling/greedy process.
    """

    def __init__(self, pattern, stop_regex, decode, vocab_size, is_greedy, prefix_length, eos_token_id, max_consider=100000):
        """ Build a new TokenHealingLogitsProcessor.

        Parameters
        ----------
        pattern : str
            The regex pattern we are seeking to match.
        stop_regex : str or list of str
            The stop regex(s) allowed to come after this pattern.
        decode : function
            The token decoding function.
        vocab_size : int
            The size of the vocabulary.
        is_greedy : bool
            The token selection mode currently in use. We need to know this so we can
            effectively take over that sampling process inside this logit processor.
        eos_token_id : int
            The end of the stop token of the model.
        max_consider : int
            How many top values to bias. Note that we could remove this option once this
            processor is performance optimized (by integrating it into the sampling/greedy process).
        """
        import torch
        
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]
        self.pattern_no_stop = regex.compile(pattern)
        self.pattern = regex.compile(pattern + "(" + "|".join(stop_regex) + ")?")
        self.decode = decode
        self.is_greedy = is_greedy
        self.prefix_length = prefix_length
        self.max_consider = max_consider
        self.bias_vector = torch.zeros(vocab_size)
        self.current_strings = None
        self.current_length = 0
        self.forced_chars = 0
        self.eos_token_id = eos_token_id

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
        to_bias = []
        for i in range(min(sort_inds.shape[1], self.max_consider)):
            proposed_string = (self.current_strings[0] + self.decode([sort_inds[0,i]]))[self.forced_chars:]
            m = self.pattern.fullmatch(proposed_string, partial=True) # partial means we don't match currently but might as the string grows
            if m:
                to_bias.append(int(sort_inds[0, i]))
                if self.is_greedy:
                    break # we are done if we are doing greedy sampling and we found the top valid hit
        
        # if we found no more valid tokens then we just end the sequence
        if not len(to_bias):
            to_bias = [self.eos_token_id]
        
        # bias allowed tokens
        min_to_bias = float(scores[0, to_bias].min())
        bias_value = scores[0, sort_inds[0, 0]] - min_to_bias + 10 # make sure the tokens that fit the pattern have higher scores than the top value
        for x in to_bias:
            self.bias_vector[x] = bias_value
        return scores + self.bias_vector.to(scores.device)

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
        self.llm = llm
        self.max_total_tokens = max_new_tokens + len(input_ids[0])
        coded_prompt = coded_prompt[:len(coded_prompt)-len(last_token_str)] # strip off the last token which will be regenerated
        self.str_pos = [len(coded_prompt) + len(self.last_token_str) for i in range(len(self.input_ids))]
        self.out_queue = queue.Queue()
        self.sequence_pos = [len(self.input_ids[0]) for i in range(len(self.input_ids))]
        self.generated_sequence = [[] for i in range(len(self.input_ids))]
        self.display_logprobs = [[] for i in range(len(self.input_ids))]
        self.generated_string = [coded_prompt for i in range(len(self.input_ids))]
        self.prefix_cache = []

    def put(self, token_obj):

        import torch
        if isinstance(token_obj, torch.Tensor):
            new_tokens = token_obj
        else:
            new_tokens = token_obj['sequences']
        

        if isinstance(new_tokens, torch.Tensor):
            new_tokens = new_tokens.cpu()
        
        # if we are given a single sequence, then make it a batch of size 1
        if len(new_tokens.shape) == 1:
            new_tokens = new_tokens.unsqueeze(0)
        
        
        # extract the scores if we are given them (and format them to be the same shape as the tokens)
        if self.logprobs:
            assert len(new_tokens) == 1, "logprobs are not supported for batched generation right now in guidance.llms.Transformers"
            new_scores = [torch.nn.functional.log_softmax(x, dim=-1).cpu() for x in token_obj['scores']]
            len_diff = len(new_tokens[0]) - len(new_scores)
            if len_diff > 0:
                new_scores = [None for i in range(len_diff)] + new_scores
            new_scores = [new_scores]
        
        out = {"choices": [None for i in range(len(self.input_ids))]}
        put_data = False
        for i in range(len(self.input_ids)):
            self.generated_sequence[i].extend(list(new_tokens[i]))
            
            # save logprobs if needed
            if self.logprobs:
                for scores in new_scores[i]:
                    if scores is None:
                        self.display_logprobs[i].append(None)
                    else:
                        top_inds = scores[0].argsort(descending=True)[:self.logprobs] # TODO: verify the [0] is always correct
                        self.display_logprobs[i].append({self.llm.id_to_token(j): float(scores[0][j]) for j in top_inds})

            if self.sequence_pos[i] < len(self.generated_sequence[i]):
                display_tokens = list(self.generated_sequence[i][self.sequence_pos[i]:])
                val = self.llm.decode(display_tokens)#[self.llm._prefix_token_id] + display_tokens)[len(self.llm._prefix_token):]
                self.generated_string[i] += val
                
                if self.str_pos[i] < len(self.generated_string[i]):
                    val = self.generated_string[i][self.str_pos[i]:]
                    finish_reason = None
                    
                    # check why we stopped
                    stop_pos = len(val) + 1
                    if len(self.generated_sequence[i]) >= self.max_total_tokens:
                        finish_reason = "length"
                    elif self.generated_sequence[i][-1] == self.llm._tokenizer.eos_token_id:
                        finish_reason = "endoftext"
                        stop_pos = len(val) - len(self.llm.decode([self.llm._tokenizer.eos_token_id]))

                    # trim off the stop regex matches if needed
                    found_partial = False
                    stop_text = None
                    if self.stop_regex is not None:# and (finish_reason is None or len(self.input_ids) > 1):
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
                                        stop_text = val[span[0]:span[1]]
                                        stop_pos = min(span[0], stop_pos)
                                        break

                    # record the reason we stopped (if we have stopped)
                    if stop_pos <= len(val):
                        finish_reason = "stop"
                    
                    # emit the data if we are not potentially in the middle of a stop sequence
                    if not found_partial or finish_reason is not None:
                        out["choices"][i] = {
                            "text": val[:stop_pos],
                            "finish_reason": finish_reason,
                            "stop_text": stop_text,
                            "logprobs": {
                                "token_healing_prefix": self.last_token_str,
                                "top_logprobs": self.display_logprobs[i][self.sequence_pos[i]:]
                            }
                        }
                        self.str_pos[i] = len(self.generated_string[i])
                        put_data = True
                self.sequence_pos[i] = len(self.generated_sequence[i])
        
        if put_data:
            self.out_queue.put(out)

    def end(self):

        # make sure we have flushed all of the data
        for i in range(len(self.input_ids)):
            assert self.str_pos[i] >= len(self.generated_string[i]), "Not all data was flushed, this means generation stopped for an unknown reason!"
        
        self.out_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.out_queue.get(timeout=self.timeout)
        if value is None:
            raise StopIteration()
        else:
            return value
