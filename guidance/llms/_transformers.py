import os
import time
import collections
import regex
import pygtrie
import queue
import threading
import collections.abc
from ._llm import LLM, LLMSession, SyncSession


class Transformers(LLM):
    """ A HuggingFace transformers language model with Guidance support.
    """

    llm_name: str = "transformers"

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

        self.model_obj, self.tokenizer = self._model_and_tokenizer(model, tokenizer, **kwargs)

        self.model_name = model if isinstance(model, str) else model.__class__.__name__
        self.caching = caching
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.token_healing = token_healing
        self.acceleration = acceleration
        if device is not None: # set the device if requested
            self.model_obj = self.model_obj.to(device)
        self.device = self.model_obj.device # otherwise note the current device

        self._token_prefix_map = self._build_token_prefix_map(model)

    def new_string_builder(self, starting_ids=None):
        return TransformersStringBuilder(self.tokenizer, starting_ids)

    def prefix_matches(self, prefix):
        """ Return the list of tokens that match the given prefix.
        """
        return [v for arr in self._token_prefix_map.values(prefix=prefix) for v in arr]

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string, **kwargs)
        
    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)
    
    def id_to_token(self, id):
        return self.tokenizer.convert_ids_to_tokens([id])[0]
    
    def token_to_id(self, token):
        return self.tokenizer.convert_tokens_to_ids([token])[0]

    def end_of_text(self):
        return self.tokenizer.eos_token

    @staticmethod
    def role_start(role):
        raise NotImplementedError("In order to use chat role tags you need to use a chat-specific subclass of Transformers for your LLM from guidance.transformers.*!")

    def _build_token_prefix_map(self, model_name):
        """ Build a map from token to index.
        """
        token_map = pygtrie.CharTrie()
        for i in range(self.tokenizer.vocab_size):
            s = self.id_to_token(i)
            if s in token_map:
                token_map[s].append(i) # handle duplicate token encodings... (GPT2 BPE has this oddly enough)
            else:
                token_map[s] = [i]

        return token_map

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
    
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,
                       top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=False,
                       cache_seed=0, caching=None, **generate_kwargs):
        """ Generate a completion of the given prompt.
        """
        
        # fill in defaults
        if temperature is None:
            temperature = self.llm.temperature
        if token_healing is None:
            token_healing = self.llm.token_healing

        # generate the cache key
        cache_params = self._cache_params(locals().copy())
        llm_cache = self.llm.cache
        key = llm_cache.create_key(self.llm.llm_name, **cache_params)

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
        stop_regex.append(regex.escape(self.llm.tokenizer.eos_token)) # make sure the end of sequence token is always included

        # handle function calling
        if "function_call" in generate_kwargs:
            assert generate_kwargs["function_call"] in ["none"], "Transformers does not yet have function call support!"
            del generate_kwargs["function_call"]

        # handle caching
        in_cache = key in llm_cache
        not_caching = (caching is not True and not self.llm.caching) or caching is False
        if not in_cache or not_caching:
            import transformers

            assert prompt != "", "You must provide a non-zero length prompt to the Transformers language model!"

            # encode the prompt
            import torch
            # encoded2 = self.llm.encode([prompt for _ in range(n)], return_tensors="pt")
            encoded = self.llm.encode(prompt)
            encoded = torch.tensor([encoded for _ in range(n)])
            if self.llm.device is not None:
                encoded = encoded.to(self.llm.device)
            input_ids = encoded#["input_ids"]
            # attention_mask = encoded["attention_mask"]
            model_config = self.llm.model_obj.config

            # ensure that we are extending a common sequence batch (our token healing assumes this right now)
            assert (input_ids[0,-1] == input_ids[:,-1]).all(), "The current token healing implementation assumes that batches are reps of the same sequence!"

            healed_token_ids = []
            processors = []
            stoppers = []

            # save what the prompt looks like when coded and then decoded (this captures added start tokens, etc.)
            coded_prompt = self.llm.decode(input_ids[0])

            # setup token healing
            if token_healing:
                healer = TokenHealingLogitsProcessor(self.llm, model_config.vocab_size, input_ids[0])
                healed_token_ids = healer.healed_token_ids
                if len(healed_token_ids) > 0:
                    input_ids = input_ids[:,:-len(healed_token_ids)]
                    # attention_mask = attention_mask[:,:-len(healed_token_ids)]
                    max_tokens += len(healed_token_ids) # increase to account for the tokens we regen for token healing
                    processors.append(healer)

            # setup logit biasing
            if logit_bias is not None:
                processors.append(BiasLogitsProcessor(self.llm, model_config.vocab_size, logit_bias))

            # find the max context length
            possible_attributes = ["max_sequence_length", "max_seq_len", "model_max_length", "n_positions", "max_position_embeddings"]
            max_context = None
            for obj in [model_config, self.llm.tokenizer]:
                for attr in possible_attributes:
                    if max_context is None:
                        max_context = getattr(obj, attr, None)
                    else:
                        break
            assert max_context is not None, "Could not find a max context length for the model! Tried: "+", ".join(possible_attributes)

            # make sure we don't run off the end of the model
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

            # add support for pattern guidance
            if pattern is not None:
                processors.append(RegexLogitsProcessor(pattern, stop_regex, self.llm, model_config.vocab_size, temperature == 0, len(coded_prompt), self.llm.tokenizer.eos_token_id))

            if stop_regex is not None:
                stoppers.append(RegexStoppingCriteria(stop_regex, self.llm, len(coded_prompt)))

            # a streamer to handle potentially partial output
            streamer = TransformersStreamer(
                input_ids=input_ids,
                stop_regex=stop_regex,
                healed_token_ids=healed_token_ids,
                prefix_length=len(coded_prompt),
                llm=self.llm,
                max_new_tokens=max_tokens,
                logprobs=logprobs
            )

            # the args for the transformers generate call
            generate_args = dict(
                inputs=input_ids,
                # attention_mask=attention_mask,
                # position_ids=position_ids,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_p=top_p,
                pad_token_id=model_config.pad_token_id if model_config.pad_token_id is not None else self.llm.tokenizer.eos_token_id,
                logits_processor=transformers.LogitsProcessorList(processors),
                stopping_criteria=transformers.StoppingCriteriaList(stoppers),
                # past_key_values=self._past_key_values,
                output_scores=logprobs is not None and logprobs > 0,
                return_dict_in_generate=True,
                **generate_kwargs
            )

            # override the model config for do_sample when the temperature requires it
            do_sample = getattr(model_config, "do_sample", None)
            if do_sample is True and temperature == 0:
                generate_args["do_sample"] = False
            elif do_sample is False and temperature > 0:
                generate_args["do_sample"] = True

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
        return llm_cache[key]
    
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

    When we tokenize the prompt the last token(s) we get are not the last token(s) we would
    have gotten if the prompt + generation was concatented and then tokenized. This
    is not good because it does not align with the pretraining of the model, so
    we "heal" this boundary by backing up as many tokens as needed and then forcing the first tokens
    generated to start with the prefix of the tokens we removed from the prompt. This could
    result in the same tokens at the end of the prompt, or some suffix of the tokens we removed
    could be replaced by a single longer one that crosses the prompt boundary.
    """

    def __init__(self, model, vocab_size, prompt_ids, bias_value=100.):
        """ Build a new TokenHealingLogitsProcessor.

        Note that bias_value is in score space (log-odds normally) and should be
        enough to ensure those tokens are the only ones used.
        """

        # loop backwards through the prompt tokens looking for places where there are possible
        # extensions that cross the prompt boundary
        prefix_str = ""
        self.extension_tokens = []
        for i in range(len(prompt_ids)-1, max(len(prompt_ids)-10, -1), -1):
            token_str = model.id_to_token(prompt_ids[i])
            prefix_str = token_str + prefix_str
            try:
                extensions = model.prefix_matches(prefix_str)
            except KeyError: # this must be a special token outside the vocab, so we assume it does not have any valid extensions
                extensions = []
            self.extension_tokens.append(extensions)
            if i != len(prompt_ids)-1:
                self.extension_tokens[-1].append(prompt_ids[i]) # add the token used in the input prompt to the list of possible extensions
        self.extension_tokens = self.extension_tokens[::-1]

        # prune off any extension token positions that don't have multiple multiple possible extensions
        found_extensions = False
        for i in range(len(self.extension_tokens)):
            if len(self.extension_tokens[i]) > 1:
                self.extension_tokens = self.extension_tokens[i:]
                found_extensions = True
                break
        if found_extensions:
            self.healed_token_ids = prompt_ids[len(prompt_ids)-len(self.extension_tokens):]
        else:
            self.extension_tokens = []
            self.healed_token_ids = []
        
        # if we have multiple possible completions past the last token, then biasing is needed
        if len(self.extension_tokens) > 0:
            import torch

            # build a set of masks for each possible extension position
            self.token_masks = []
            for i in range(len(self.extension_tokens)):
                token_mask = torch.zeros(vocab_size)
                token_mask.scatter_(0, torch.tensor(self.extension_tokens[i]), bias_value)
                if model.device is not None:
                    token_mask = token_mask.to(model.device)
                self.token_masks.append(token_mask)

        self.num_extensions = 0

    def __call__(self, input_ids, scores):

        # we only bias the first token generated
        if self.num_extensions >= len(self.extension_tokens):
            return scores
        self.num_extensions += 1

        # check if the last token was from the original prompt (if not then we have already "healed" by choosing a token that crosses the prompt boundary)
        if self.num_extensions > 1 and input_ids[0][-1] != self.healed_token_ids[self.num_extensions-2]:
            return scores

        # handle list inputs
        if isinstance(scores, list):
            import torch
            scores = torch.tensor(scores)

        # make only allowed tokens possible
        return scores + self.token_masks[self.num_extensions-1]
    
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

        # handle list inputs
        if isinstance(scores, list):
            import torch
            scores = torch.tensor(scores)

        return scores + self.bias_vector
    
class RegexLogitsProcessor():
    """ Pattern guiding.
    
    Guide generation to match a regular expression.
    TODO: currently slow, could be made much faster by doing rejection sampling inline with the sampling/greedy process.
    """

    def __init__(self, pattern, stop_regex, llm, vocab_size, is_greedy, prefix_length, eos_token_id, max_consider=500000):
        """ Build a new TokenHealingLogitsProcessor.

        Parameters
        ----------
        pattern : str
            The regex pattern we are seeking to match.
        stop_regex : str or list of str
            The stop regex(s) allowed to come after this pattern.
        llm : function
            The llm.
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
        self.llm = llm
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

        # handle 1D inputs
        one_dim = False
        if not isinstance(input_ids[0], collections.abc.Sequence) and not (hasattr(input_ids[0], "shape") and len(input_ids[0].shape) > 0):
            one_dim = True
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            scores = torch.tensor(scores).unsqueeze(0)

        # extend our current strings
        if self.current_strings is None:
            self.current_strings = [self.llm.new_string_builder() for i in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i].extend(input_ids[i][self.current_length:])

        assert len(self.current_strings) == 1, "Regex patterns guides do not support batched inference with Transformers yet!"

        self.current_length = len(input_ids[0])
        
        # compute the bias values
        self.bias_vector[:] = 0
        sort_inds = torch.argsort(scores, 1, True)
        to_bias = []
        for i in range(min(sort_inds.shape[1], self.max_consider)):
            self.current_strings[0].extend([sort_inds[0,i]])
            proposed_string = str(self.current_strings[0])[self.prefix_length:]
            self.current_strings[0].pop()
            m = self.pattern.fullmatch(proposed_string, partial=True) # partial means we don't match currently but might as the string grows
            if m:
                to_bias.append(int(sort_inds[0, i]))
                if self.is_greedy: # TODO: make this much faster for non-greedy sampling (by tracking how much prob mass we have looked through perhaps...)
                    break # we are done if we are doing greedy sampling and we found the top valid hit
        
        # if we found no more valid tokens then we just end the sequence
        if not len(to_bias):
            to_bias = [self.eos_token_id]
        
        # bias allowed tokens
        min_to_bias = float(scores[0, to_bias].min())
        bias_value = scores[0, sort_inds[0, 0]] - min_to_bias + 10 # make sure the tokens that fit the pattern have higher scores than the top value
        for x in to_bias:
            self.bias_vector[x] = bias_value
        out = scores + self.bias_vector.to(scores.device)
        if one_dim:
            return out[0]
        else:
            return out

class RegexStoppingCriteria():
    def __init__(self, stop_pattern, llm, prefix_length):
        if isinstance(stop_pattern, str):
            self.stop_patterns = [regex.compile(stop_pattern)]
        else:
            self.stop_patterns = [regex.compile(pattern) for pattern in stop_pattern]
        self.prefix_length = prefix_length
        self.llm = llm
        self.current_strings = None
        self.current_length = 0

    def __call__(self, input_ids, scores, **kwargs):

        # handle 1D inputs
        if not isinstance(input_ids[0], collections.abc.Sequence) and not (hasattr(input_ids[0], "shape") and len(input_ids[0].shape) > 0):
            input_ids = [input_ids]

        # extend our current strings
        if self.current_strings is None:
            self.current_strings = [self.llm.new_string_builder() for _ in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i].extend(input_ids[i][self.current_length:])
        
        self.current_length = len(input_ids[0])
        
        # check if all of the strings match a stop string (and hence we can stop the batch inference)
        all_done = True
        for i in range(len(self.current_strings)):
            found = False
            for s in self.stop_patterns:
                if s.search(str(self.current_strings[i])[self.prefix_length:]):
                    found = True
            if not found:
                all_done = False
                break
        
        return all_done

class TransformersStringBuilder():
    """This deals with the complexity of building up a string from tokens bit by bit."""
    def __init__(self, tokenizer, starting_ids=None):
        self.tokenizer = tokenizer
        self.token_strings = []
        self._joint_string = ""
        if starting_ids is not None:
            self.extend(starting_ids)

    def extend(self, new_ids):
        new_token_strings = self.tokenizer.convert_ids_to_tokens(new_ids)
        self.token_strings.extend(new_token_strings)
        new_str = self.tokenizer.convert_tokens_to_string(self.token_strings)
        diff_str = new_str[len(self._joint_string):]
        self._joint_string = new_str
        return diff_str

    def pop(self):
        """Remove the last token from the string and return text it removed."""
        self.token_strings.pop()
        new_str = self.tokenizer.convert_tokens_to_string(self.token_strings)
        diff_str = self._joint_string[len(new_str):]
        self._joint_string = new_str
        return diff_str

    def __str__(self):
        return self._joint_string

    def __len__(self):
        return len(self._joint_string)

class TransformersStreamer():
    def __init__(self, input_ids, stop_regex, healed_token_ids, prefix_length, llm, max_new_tokens, logprobs, timeout=None):

        self.input_ids = input_ids
        self.stop_regex = stop_regex
        self.healed_token_ids = healed_token_ids
        self.logprobs = logprobs
        self.llm = llm
        self.max_total_tokens = max_new_tokens + len(input_ids[0])
        self.timeout = timeout
        self.str_pos = [prefix_length for i in range(len(self.input_ids))]
        self.out_queue = queue.Queue()
        self.sequence_pos = [len(self.input_ids[0]) for i in range(len(self.input_ids))]
        self.generated_sequence = [[] for i in range(len(self.input_ids))]
        self.display_logprobs = [[] for i in range(len(self.input_ids))]
        self.generated_string = [self.llm.new_string_builder(input_ids[0]) for i in range(len(self.input_ids))]
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
                val = self.generated_string[i].extend(display_tokens)
                # val = self.llm.decode(display_tokens)#[self.llm._prefix_token_id] + display_tokens)[len(self.llm._prefix_token):]
                # self.generated_string[i] += val
                
                if self.str_pos[i] < len(self.generated_string[i]):
                    val = str(self.generated_string[i])[self.str_pos[i]:]
                    finish_reason = None
                    
                    # check why we stopped
                    stop_pos = len(val) + 1
                    if len(self.generated_sequence[i]) >= self.max_total_tokens:
                        finish_reason = "length"
                    elif self.generated_sequence[i][-1] == self.llm.tokenizer.eos_token_id:
                        finish_reason = "endoftext"
                        eos_str = self.generated_string[i].pop() # remove the end of text token
                        stop_pos = len(val) - len(eos_str)

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
                                # "token_healing_prefix": self.last_token_str,
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
