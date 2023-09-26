import torch
import regex
import numpy as np
import os
import time
import itertools
import numpy as np
import collections
from .._utils import Trie
from ._model import Model

class MatchMock():
    __slots__ = ('partial')
    def groupdict(self):
        return self
PARTIAL_MATCH = MatchMock()
PARTIAL_MATCH.partial = True

class Local(Model):
    def __init__(self, tokens, bos_token_id, eos_token_id=None, echo=True):
        super().__init__(echo)
        
        self.tokens = tokens
        self.bos_token_id = bos_token_id
        self.bos_token = self.tokens[self.bos_token_id]
        self.eos_token_id = eos_token_id if eos_token_id is not None else bos_token_id
        self.eos_token = self.tokens[self.eos_token_id]

        # build a prefix tree of the tokens
        self._token_trie = Trie(tokens, np.arange(len(tokens)))
        self._token_trie.match = True
        self._token_trie.match_version = 0

        # all mutable state goes in the cache dictionary
        self._cache_state["cache_token_ids"] = []
        self._cache_state["new_token_ids"] = []

    def _get_logits(self):
        '''A fake method designed to be overriden by subclasses.'''
        self._cache_state["cache_token_ids"].extend(self._new_token_ids)
        self.self._new_token_ids = []

        # pretend to extend the KV cache and update the log probs
        return torch.randn(len(self.tokens))

    def _longest_token_match(self, string):
        '''Greedy token matching.'''
        if string.startswith("\n"):
            pass
        trie_pos = self._token_trie
        for i,c in enumerate(string):
            if c in trie_pos.children:
                trie_pos = trie_pos.children[c]
            else:
                return string[:i], trie_pos.value # note that if there are redudant tokens we choose the one stored in the trie
        if len(trie_pos.children) == 0:
            return string[:i+1], trie_pos.value
        else:
            return None,None # more than one token can match this string

    def __call__(self, pattern, max_tokens=100, n=1, top_p=1, temperature=0.0, ensure_bos_token=True):
        prompt = str(self)

        # add the beginning of sequence token if needed
        if ensure_bos_token and not prompt.startswith(self.bos_token):
            prompt = self.bos_token + prompt

        # find out how much of the prompt we have in the KV cache (and strip from the prompt what we already have)
        cache_token_ids = self._cache_state["cache_token_ids"]
        if len(cache_token_ids) > 0:
            cache_pos = 0
            for i,id in enumerate(cache_token_ids):
                token = self.tokens[id]
                if prompt[cache_pos:cache_pos+len(token)] != token:
                    self._cache_state["cache_token_ids"] = cache_token_ids[:i]
                    break
                cache_pos += len(token)
            # self._cache_tokens = self._cache_tokens[:i]
            prompt = prompt[cache_pos:]
        self._cache_state["new_token_ids"] = []

        # send the known prompt tokens to the KV cache as a batch
        forced_token_ids = []
        while True:
            token, token_id = self._longest_token_match(prompt)
            if token_id is not None:
                forced_token_ids.append(token_id)
                prompt = prompt[len(token):]
            else:
                break
        if len(forced_token_ids) > 0:
            self._cache_state["new_token_ids"].extend(forced_token_ids)
        
        # move whatever is not cached from the prompt into the pattern (since the pattern is what we will generate)
        # note we also anchor the pattern to the start of the sequence
        pattern = "^" + regex.escape(prompt, literal_spaces=True) + pattern
        pattern_obj = regex.compile(pattern, flags=regex.DOTALL)
        const_prefix_len = len(pattern_obj._pickled_data[-3])

        assert n == 1, "Still need to add support for n > 1!"

        extracted_stop_pattern = regex.compile("(" + pattern[pattern.index("(?P<stop>")+9:-1] + ")$", flags=regex.DOTALL)

        # loop until we have generated a complete pattern
        # call_count = [0]
        hidden_count = len(prompt) # we don't emit the part of the prompt we have to regenerate for token healing
        generated_text = ""
        delayed_text = ""
        for token_count in range(max_tokens):

            # TODO: eventually we could try and check if the regex "forces" the next token so we can skip
            #       logit computation entirely. This might only make sense if we can make the regex matching
            #       really fast (integrate with the FSM directly) or make it report when a character is forced.

            # compute the order in which we prefer the tokens
            logits = self._get_logits()
            if temperature == 0:
                sampling_order = torch.argsort(logits, descending=True).cpu().numpy() # we need numpy so the enumerate below does not get really slow...
            else:
                assert top_p == 1, "Still need to add support for top_p!"
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                sampling_order = torch.multinomial(probs, len(probs)).cpu().numpy()

            # find the best allowed token
            #call_count[0] = 0
            self._token_trie.match_version += 1
            gen_len = len(generated_text) + len(delayed_text)
            for i,sampled_token_ind in enumerate(sampling_order):
                sampled_token = self.tokens[sampled_token_ind]

                # check to see if the sampled token is allowed (TODO: consider if this needs more optimized...python loops are slow)
                token_pos = 1
                node = self._token_trie.children[sampled_token[0]]
                while True:
                    if node.match_version < self._token_trie.match_version:
                        node.match_version = self._token_trie.match_version
                        if gen_len + token_pos <= const_prefix_len:
                            if pattern[gen_len + token_pos] == sampled_token[token_pos-1]:
                                node.match = PARTIAL_MATCH
                            else:
                                node.match = None
                        else:
                            #call_count[0] += 1
                            m = pattern_obj.match(generated_text+delayed_text+sampled_token[:token_pos], partial=True)
                            node.match = m
                    
                    if token_pos == len(sampled_token):
                        m = node.match
                        break
                    else:
                        if node.match:
                            token_pos += 1
                            node = node.children[sampled_token[token_pos-1]]
                        else:
                            m = None
                            break
                
                if m is not None:
                    break
            assert m is not None, f"There were no tokens found that could encode: `{pattern[gen_len + token_pos]}`, perhaps the model vocabulary does not contain this token?"
            #print("call_count", call_count[0], "`" + sampled_token + "`", i)
            
            # delay emitting if we might be starting the stop pattern
            new_text = delayed_text + sampled_token
            delayed_text = ""
            stop_match = extracted_stop_pattern.search(generated_text + new_text, partial=True)
            if stop_match and stop_match.end() - stop_match.start() > 0:

                # emit delayed text before the match start
                if stop_match.start() > len(generated_text):
                    offset = stop_match.start() - len(generated_text)
                    delayed_text = new_text[offset:]
                    new_text = new_text[:offset]
                else:
                    delayed_text = new_text
                    new_text = ""

            # if we have a full match we are done
            if not m.partial and len(m.groupdict()["stop"]) > 0:
                new_text += delayed_text
                generated_text += new_text
                
                # strip the stop group
                stop = m.group('stop')
                if len(stop) > 0:
                    generated_text = generated_text[:-len(stop)]

                # if we exactly match the end of the pattern then we can commit to this last token 
                if m.span()[1] == len(generated_text):
                    self._cache_state["new_token_ids"].append(sampled_token_ind)
                
                if hidden_count < len(new_text) - len(stop):
                    yield sampled_token[hidden_count:len(new_text) - len(stop)], m.groupdict()
                break # we are done!
            else:
                generated_text += new_text
                
                # yeild the snippet of text created by the next token
                out = new_text[hidden_count:]
                if len(out) > 0:
                    yield out, m.groupdict()
                    hidden_count = 0
                else:
                    hidden_count -= len(new_text)

                self._cache_state["new_token_ids"].append(sampled_token_ind)