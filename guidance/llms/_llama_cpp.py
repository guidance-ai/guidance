import os
import time
import collections

import numpy as np
import regex
import pygtrie
import queue
import threading
import logging

from llama_cpp import Llama, llama_n_vocab

from ._llm import LLM, LLMSession, SyncSession


class LlamaCppSettings:
    model: str = "../ggml-model.q4_1.bin"
    n_ctx: int = 2048
    n_batch: int = 8
    n_threads: int = 4
    f16_kv: bool = True
    use_mlock: bool = True
    embedding: bool = False
    last_n_tokens_size: int = 256
    n_gpu_layers: int = 0
    logits_all: bool = True


class LlamaCpp(LLM):
    """ A HuggingFace transformers language model with Guidance support.
    """

    cache = LLM._open_cache("_llama_cpp.diskcache")

    def __init__(self, settings=LlamaCppSettings(), caching=True, token_healing=False, acceleration=False,
                 temperature=0.25,
                 role_start=None, role_end=None):
        super().__init__()
        self.settings = settings
        self.model_obj = Llama(
            settings.model,
            n_gpu_layers=settings.n_gpu_layers,
            f16_kv=settings.f16_kv,
            use_mlock=settings.use_mlock,
            embedding=settings.embedding,
            n_threads=settings.n_threads,
            n_batch=settings.n_batch,
            n_ctx=settings.n_ctx,
            last_n_tokens_size=settings.last_n_tokens_size,
            logits_all=settings.logits_all
        )
        self.device = None
        self.vocab_size = llama_n_vocab(self.model_obj.ctx)
        self._generate_call = self.model_obj.create_completion
        base_name = os.path.basename(settings.model)  # get the name of the file
        file_name_without_extension, _ = os.path.splitext(base_name)
        self.model_name = file_name_without_extension
        self.caching = caching
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.token_healing = token_healing
        self.acceleration = acceleration
        temp = self.model_obj.tokenize("temp".encode(), True)
        self._prefix_ids = [temp[0], 100]  # token ids that we use to decode tokens after a prefix
        self._prefix_str = self.model_obj.detokenize(tokens=self._prefix_ids).decode("utf-8")

        self._token_prefix_map = self._build_token_prefix_map()

    def prefix_matches(self, prefix):
        """ Return the list of tokens that match the given prefix.
        """
        return [v for arr in self._token_prefix_map.values(prefix=prefix) for v in arr]

    def encode(self, string, fragment=True, **kwargs):

        if isinstance(string, bytes):
            string = string.decode("utf-8")

        if fragment:
            string = self._prefix_str + string
        out = {}
        if "return_tensors" in kwargs:
            out['input_ids'] = self.model_obj.tokenize(string.encode("utf-8"))
        else:
            out['input_ids'] = self.model_obj.tokenize(string.encode('utf-8'))

        # remove the start token when we are encoding a suffix
        if fragment:
            if out['input_ids'][1] == self.model_obj.token_bos():  # sometime the tokenizer adds an extra start token
                out = out['input_ids'][3:]
            else:
                out = out['input_ids'][2:]

        return out

    # def role_start(self, role):
    #     """ The starting role tag for chat models.

    #     #TODO Right now this just assumes the StableLM syntax, but this should be expanded later.
    #     """
    #     return "<|"+role.upper()+"|>"

    # def role_end(self, role=None):
    #     return ""

    @staticmethod
    def role_start(role):
        raise NotImplementedError(
            "In order to use chat role tags you need to use a chat-specific subclass of Transformers for your LLM from guidance.transformers.*!")

    def decode(self, tokens, fragment=True, **kwargs):

        # if the last token is the end of string token, or the first is a start of string we remove it because it cause odd spacing decoding of fragments
        add_eos = ""
        add_bos = ""
        if fragment:
            if len(tokens) > 0 and tokens[-1] == self.model_obj.token_eos():
                add_eos = self.model_obj.detokenize([self.model_obj.token_eos()]).decode("utf-8")
                tokens = tokens[:-1]
            if len(tokens) > 0 and tokens[0] == self.model_obj.token_bos():
                add_bos = self.model_obj.detokenize([self.model_obj.token_bos()]).decode("utf-8")
                tokens = tokens[1:]

        if fragment:
            return add_bos + (self.model_obj.detokenize(self._prefix_ids + list(tokens))[
                             len(self._prefix_str):]).decode("utf-8", errors="ignore") + add_eos
        else:
            return add_bos + ( self.model_obj.detokenize(tokens).decode("utf-8", errors="ignore")) + add_eos

    def _build_token_prefix_map(self):
        """ Build a map from token to index.
        """
        token_map = pygtrie.CharTrie()
        for i in range(self.vocab_size):
            s = self.decode([i])
            if s in token_map:
                token_map[s].append(i)  # handle duplicate token encodings... (GPT2 BPE has this oddly enough)
            else:
                token_map[s] = [i]

        return token_map

    def session(self, asynchronous=False):
        if asynchronous:
            return LlamaCppSession(self)
        else:
            return SyncSession(LlamaCppSession(self))


class LlamaCppSession(LLMSession):
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
                        input_ids = input_ids[:, len(self._prefix_cache):]
                        # if "attention_mask" in kwargs:
                        #     kwargs["attention_mask"] = kwargs["attention_mask"][:,len(self._prefix_cache):]
                        model_kwargs = method(input_ids, **kwargs)

                        # provide the past key values for the actual model call
                        model_kwargs["past_key_values"] = self._past_key_values
                        model_kwargs["position_ids"] = model_kwargs["position_ids"][:,
                                                       len(self._prefix_cache):]  # and update position ids

                        # we only need to do this first time, after that the past key values will
                        # be up until the last token, just like transformer models normally expect
                        # so we can clear our cache and let transformers cache like normal
                        self._prefix_cache = []  # this will get refilled once the generate call is done

                        return model_kwargs
                    else:
                        return method(input_ids, **kwargs)

                decorate_prep_step.__func__ = method.__func__  # make us still look like a bound method
                return decorate_prep_step

            if getattr(self.llm.model_obj, "_orig_prepare_method", None) is None:
                self.llm.model_obj._orig_prepare_method = self.llm.model_obj.tokenize
            self.llm.model_obj.prepare_inputs_for_generation = prep_step_decorator(
                self.llm.model_obj._orig_prepare_method)

            # decorate the update step to save the past key values
            def update_step_decorator(method):
                def decorate_update_step(outputs, *args, **kwargs):
                    # save the past key values
                    self._past_key_values = outputs.past_key_values

                    return method(outputs, *args, **kwargs)

                return decorate_update_step

            if getattr(self.llm.model_obj, "_orig_update_method", None) is None:
                self.llm.model_obj._orig_update_method = None
            self.llm.model_obj._update_model_kwargs_for_generation = update_step_decorator(
                self.llm.model_obj._orig_update_method)

        return self

    # def __call__(self, *args, **kwargs):
    #     return self.__call__(*args, **kwargs)

    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,
                       top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=False,
                       cache_seed=0, caching=None):
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
        stop_regex.append(
            regex.escape("</s>"))  # make sure the end of sequence token is always included

        # handle caching
        if key not in self.llm.cache or (caching is not True and not self.llm.caching) or caching is False:
            import transformers
            # import torch
            # encode the prompt
            encoded = self.llm.encode(prompt, return_tensors="pt", fragment=False)
            if self.llm.device is not None:
                encoded = encoded.to(self.llm.device)
            input_ids = encoded["input_ids"]
            model_config = self.llm.settings

            # ensure that we are extending a common sequence batch (our token healing assumes this right now)
            # assert (input_ids[0, -1] == input_ids[:,
            #                             -1]).all(), "The current token healing implementation assumes that batches are reps of the same sequence!"

            last_token_str = ""
            processors = []
            stoppers = []

            # save what the prompt looks like when coded and then decoded (this captures added start tokens, etc.)
            coded_prompt = self.llm.decode(input_ids)

            # setup token healing
            if token_healing:
                # pop off the last token since we will regen it
                last_token_id = input_ids[-1]
                last_token_str = self.llm.decode([last_token_id])
                healer = TokenHealingLogitsProcessor(self.llm, self.llm.vocab_size, last_token_str)
                if healer.should_bias:
                    input_ids = input_ids[0:-1]
                    max_tokens += 1  # add one for the token we regen for token healing
                    processors.append(healer)
                else:
                    last_token_str = ""

            # setup logit biasing
            if logit_bias is not None:
                processors.append(BiasLogitsProcessor(self.llm, self.llm.vocab_size, logit_bias))

            # make sure we don't run off the end of the model
            max_context = (getattr(model_config, "n_ctx", None) or getattr(model_config, "max_seq_len",
                                                                           None) or getattr(model_config,
                                                                                            "n_positions",
                                                                                            None) or getattr(
                model_config, "max_position_embeddings"))
            if max_tokens + len(input_ids) > max_context:
                max_tokens = max_context - len(input_ids)

            # find how much of the prompt is cached
            for prefix_match_len, token in enumerate(input_ids):
                if prefix_match_len >= len(self._prefix_cache) or token != self._prefix_cache[prefix_match_len]:
                    break

            # trim the cache to what we can use
            if prefix_match_len > 0 and prefix_match_len < len(self._prefix_cache):
                self._past_key_values = tuple(
                    (key[:, :, :prefix_match_len, :], value[:, :, :prefix_match_len, :]) for key, value in
                    self._past_key_values)  # TODO: this is specific to the GPT2 tensor layout
                self._prefix_cache = self._prefix_cache[:prefix_match_len]

            # see if we need to returns the scores
            # output_scores = logprobs is not None and logprobs > 0

            # position_ids = torch.arange(prefix_match_len, input_ids.shape[-1], dtype=torch.long).unsqueeze(0)

            # trim input ids that we will pull from the cache instead of computing keys and values for
            # input_ids = input_ids[:,prefix_match_len:]

            # add support for pattern guidance
            if pattern is not None:
                processors.append(RegexLogitsProcessor(pattern, stop_regex, self.llm.decode, self.llm.vocab_size,
                                                       False, len(coded_prompt),
                                                       self.llm.model_obj.token_eos()))

            if stop_regex is not None:
                stoppers.append(RegexStoppingCriteria(stop_regex, self.llm.decode, len(coded_prompt)))

            # a streamer to handle potentially partial output
            streamer = LlamaCppStreamer(
                input_ids=input_ids,
                stop_regex=stop_regex,
                last_token_str=last_token_str,
                coded_prompt=coded_prompt,
                llm=self.llm,
                max_new_tokens=max_tokens,
                logprobs=logprobs,
                timeout=10
            )
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            # the args for the transformers generate call
            generate_args = dict(
                prompt=prompt,
                # position_ids=position_ids,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=[],
                # pad_token_id=self.llm.model_obj.token_eos(),
                logprobs=logprobs,
                logits_processors=processors,
                stopping_criterias=stoppers,
                # past_key_values=self._past_key_values,
                # output_scores=logprobs is not None and logprobs > 0,
                # return_dict_in_generate=True
            )

            # override the model config for do_sample when the temperature requires it
            do_sample = getattr(self.llm.settings, "do_sample", None)
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
            self._prefix_cache = streamer.generated_sequence[:self._past_key_values[0][0].shape[
                2]]  # self._past_key_values is already saved, this just aligns with it

    def _stream_then_save(self, streamer, key, thread):
        list_out = []
        for out in streamer:
            list_out.append(out)
            yield out
        thread.join()  # clean up the thread
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


class LlamaCppStreamer():
    def __init__(self, input_ids, stop_regex, last_token_str, coded_prompt, llm, max_new_tokens, logprobs,
                 timeout=None):
        self.timeout = timeout
        self.input_ids = input_ids
        self.stop_regex = stop_regex
        self.logprobs = logprobs
        self.last_token_str = last_token_str
        # self.coded_prompt = coded_prompt
        self.llm = llm
        self.max_total_tokens = max_new_tokens + len(input_ids)
        coded_prompt = coded_prompt[:len(coded_prompt) - len(
            last_token_str)]  # strip off the last token which will be regenerated
        self.str_pos = len(coded_prompt) + len(self.last_token_str)
        self.out_queue = queue.Queue()
        self.sequence_pos = 0
        self.generated_sequence = []
        self.generated_scores = []
        self.generated_string = coded_prompt
        self.prefix_cache = []

    def put(self, token_obj):

        import torch
        if isinstance(token_obj, torch.Tensor):
            new_tokens = token_obj
        else:
            new_tokens = token_obj['choices'][0]['text']

        # extract the scores if we are given them (and format them to be the same shape as the tokens)
        if self.logprobs:
            new_scores = token_obj['choices'][0]['logprobs']['token_logprobs']
            len_diff = len(new_tokens) - len(new_scores)
            if len_diff > 0:
                new_scores = [0.0 for i in range(len_diff)] + new_scores
            new_scores = [new_scores]

        out = {"choices": []}
        put_data = False
        self.generated_sequence.extend(new_tokens)
        if self.logprobs:
            self.generated_scores.extend(list(new_scores))

        if self.sequence_pos < len(self.generated_sequence):
            val = new_tokens  # [self.llm._prefix_token_id] + display_tokens)[len(self.llm._prefix_token):]
            self.generated_string += val

            if self.str_pos < len(self.generated_string):

                display_logprobs = None
                if self.logprobs:
                    display_logprobs = token_obj['choices'][0]['logprobs']['top_logprobs']

                val = self.generated_string[self.str_pos:]
                finish_reason = None

                # check why we stopped
                stop_pos = len(val) + 1
                if len(self.generated_sequence) >= self.max_total_tokens:
                    finish_reason = "length"
                elif self.generated_sequence[-1] == self.llm.model_obj.token_eos():
                    finish_reason = "endoftext"
                    stop_pos = len(val) - len(self.llm.decode([self.llm.model_obj.token_eos()]))

                # trim off the stop regex matches if needed
                found_partial = False
                if self.stop_regex is not None and (finish_reason is None or len(self.input_ids) > 1):
                    stop_regex_obj = [regex.compile(s) for s in self.stop_regex]
                    for s in stop_regex_obj:
                        m = s.search(val)
                        if m:
                            span = m.span()
                            if span[1] > span[0]:
                                if m.partial:  # we might be starting a stop sequence, so we can't emit anything yet
                                    found_partial = True
                                    break
                                else:
                                    stop_pos = min(span[0], stop_pos)

                # record the reason we stopped (if we have stopped)
                if stop_pos <= len(val):
                    finish_reason = "stop"

                if not found_partial:
                    out["choices"].append({
                        "text": val[:stop_pos],
                        "finish_reason": finish_reason,
                        "logprobs": {"token_healing_prefix": self.last_token_str,
                                     "top_logprobs": display_logprobs}
                    })
                    self.str_pos = len(self.generated_string)
                    put_data = True
            self.sequence_pos = len(self.generated_sequence)

        if put_data:
            self.out_queue.put(out)

    def end(self):

        # make sure we have flushed all of the data
        for i in range(len(self.input_ids)):
            assert self.str_pos[i] >= len(self.generated_string[
                                              i]), "Not all data was flushed, this means generation stopped for an unknown reason!"

        self.out_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            value = self.out_queue.get(timeout=self.timeout)
        except queue.Empty:
            value = None
        if value is None:
            raise StopIteration()
        else:
            return value


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
            assert len(
                allowed_first_tokens) > 0, "Error in token healing map! No match found for: `" + last_token_str + "`"
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
        import torch
        # we only bias the first token generated
        if not self.should_bias:
            return scores
        self.should_bias = False

        # make only allowed tokens possible
        return (torch.tensor(scores) + self.first_token_mask).tolist()


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
        self.bias_vector = self.bias_vector

    def __call__(self, input_ids, scores):
        import torch
        return (torch.tensor(scores) + self.bias_vector).tolist()


class RegexLogitsProcessor():
    """ Pattern guiding.

    Guide generation to match a regular expression.
    TODO: currently slow, could be made much faster by doing rejection sampling inline with the sampling/greedy process.
    """

    def __init__(self, pattern, stop_regex, decode, vocab_size, is_greedy, prefix_length, eos_token_id,
                 max_consider=100000):
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
            self.current_strings = ""
        self.current_strings += self.decode(input_ids[self.current_length:])


        # trim off the prefix string so we don't look for stop matches in the prompt
        if self.current_length == 0:
            self.forced_chars = self.prefix_length - len(
                self.current_strings)  # account for token healing forced prefixes
            self.current_strings = self.current_strings[self.prefix_length:]


        self.current_length = len(input_ids)

        # compute the bias values
        self.bias_vector[:] = 0

        sort_inds = torch.argsort(torch.tensor(scores), 0, True)
        to_bias = []
        for i in range(min(sort_inds.shape[0], self.max_consider)):
            proposed_string = (self.current_strings + (self.decode([sort_inds[i].item()])))[self.forced_chars:]
            m = self.pattern.fullmatch(proposed_string, partial=True)  # partial means we don't match currently but might as the string grows
            if m:
                to_bias.append(int(sort_inds[i]))
                if self.is_greedy:
                    break  # we are done if we are doing greedy sampling and we found the top valid hit

        # if we found no more valid tokens then we just end the sequence
        if not len(to_bias):
            to_bias = [self.eos_token_id]

        # bias allowed tokens
        min_to_bias = float(torch.tensor(scores)[to_bias].min())
        bias_value = torch.tensor(scores)[sort_inds[0]] - min_to_bias + 10  # make sure the tokens that fit the pattern have higher scores than the top value
        for x in to_bias:
            self.bias_vector[x] = bias_value
        import torch
        return (torch.tensor(scores) + self.bias_vector).tolist()


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
            self.current_strings = ""
        self.current_strings += self.decode(input_ids[self.current_length:])


        # trim off the prefix string so we don't look for stop matches in the prompt
        if self.current_length == 0:
            self.current_strings = self.current_strings[self.prefix_length:]


        self.current_length = len(input_ids)

        # check if all of the strings match a stop string (and hence we can stop the batch inference)
        all_done = True
        found = False
        for s in self.stop_patterns:
            if s.search(self.current_strings):
                found = True
        if not found:
            all_done = False


        return all_done