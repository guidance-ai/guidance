import torch
import regex
import numpy as np
import os
import time
import itertools
import numpy as np
import collections
from .._utils import Trie
from ._engine import Engine

class MatchMock():
    pass
PARTIAL_MATCH = MatchMock()
PARTIAL_MATCH.partial = True


class LocalEngine():
    def __init__(self, tokens, bos_token_id):
        self.tokens = tokens
        self.bos_token_id = bos_token_id
        self.bos_token = self.tokens[self.bos_token_id]
        self._cache_token_ids = []
        self._logits = torch.randn(len(tokens))
        self._match_version = 1
        self._trie = Trie(tokens, np.arange(len(tokens)))

    def extend_model(self, token_ids):
        '''A fake method designed to be overriden by subclasses.'''
        self._cache_token_ids.extend(token_ids)

        # pretend to extend the KV cache and update the log probs
        self._logits = torch.randn(len(self.tokens))

    def longest_token_match(self, string):
        '''Greedy token matching.'''
        trie_pos = self._trie
        for i,c in enumerate(string):
            if c in trie_pos.children:
                trie_pos = trie_pos.children[c]
            else:
                return string[:i], trie_pos.value # note that if there are redudant tokens we choose the one stored in the trie
        if len(trie_pos.children) == 0:
            return string[:i+1], trie_pos.value
        else:
            return None,None # more than one token can match this string
    
    def __call__(self, prompt, pattern, max_tokens=100, n=1, top_p=1, temperature=0.0, ensure_bos_token=True):

        # add the beginning of sequence token if needed
        if ensure_bos_token and not prompt.startswith(self.bos_token):
            prompt = self.bos_token + prompt

        # find out how much of the prompt we have in the KV cache (and strip from the prompt what we already have)
        if len(self._cache_token_ids) > 0:
            cache_pos = 0
            for i,id in enumerate(self._cache_token_ids):
                token = self.tokens[id]
                if prompt[cache_pos:cache_pos+len(token)] != token:
                    self._cache_token_ids = self._cache_token_ids[:i]
                    break
                cache_pos += len(token)
            # self._cache_tokens = self._cache_tokens[:i]
            prompt = prompt[cache_pos:]
            

        # send the known prompt tokens to the KV cache as a batch
        new_token_ids = []
        while True:
            token, token_id = self.longest_token_match(prompt)
            if token_id is not None:
                new_token_ids.append(token_id)
                prompt = prompt[len(token):]
            else:
                break
        if len(new_token_ids) > 0:
            self.extend_model(new_token_ids)
        
        # move whatever is not cached from the prompt into the pattern (since the pattern is what we will generate)
        # note we also anchor the pattern to the start of the sequence
        pattern = "^" + regex.escape(prompt, literal_spaces=True) + pattern
        pattern_obj = regex.compile(pattern)
        const_prefix_len = len(pattern_obj._pickled_data[-3])

        assert n == 1, "Still need to add support for n > 1!"

        # loop until we have generated a complete pattern
        # call_count = [0]
        hidden_count = len(prompt) # we don't emit the part of the prompt we have to regenerate for token healing
        generated_text = ""
        for token_count in range(max_tokens):

            # TODO: eventually we could try and check if the regex "forces" the next token so we can skip
            #       logit computation entirely. This might only make sense if we can make the regex matching
            #       really fast (integrate with the FSM directly) or make it report when a character is forced.

            # compute the order in which we prefer the tokens
            if temperature == 0:
                sampling_order = torch.argsort(self._logits, descending=True).cpu().numpy() # we need numpy so the enumerate below does not get really slow...
            else:
                assert top_p == 1, "Still need to add support for top_p!"
                probs = torch.nn.functional.softmax(self._logits / temperature, dim=-1)
                sampling_order = torch.multinomial(probs, len(probs)).cpu().numpy()

            # find the best allowed token
            #call_count[0] = 0
            self._match_version += 1
            gen_len = len(generated_text)
            for i,sampled_token_ind in enumerate(sampling_order):
                sampled_token = self.tokens[sampled_token_ind]

                # check to see if the sampled token is allowed (TODO: consider if this needs more optimized...python loops are slow)
                token_pos = 1
                node = self._trie.children[sampled_token[0]]
                while True:
                    if node.match_version < self._match_version:
                        node.match_version = self._match_version
                        if gen_len + token_pos <= const_prefix_len:
                            if pattern[gen_len + token_pos] == sampled_token[token_pos-1]:
                                node.match = PARTIAL_MATCH
                            else:
                                node.match = None
                        else:
                            #call_count[0] += 1
                            m = pattern_obj.match(generated_text+sampled_token[:token_pos], partial=True)
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
            #print("call_count", call_count[0], "`" + sampled_token + "`", i)
            generated_text += sampled_token

            # if we have a full match we are done
            if not m.partial:
                end = m.span()[1] - len(generated_text) + len(sampled_token)

                # if we exactly match the end of the pattern then we can commit to this last token 
                if end == len(sampled_token):
                    self.extend_model([sampled_token_ind])
                
                if hidden_count < end:
                    yield sampled_token[hidden_count:end], m.groupdict()
                break # we are done!
            else:

                # yeild the snippet of text created by the next token
                out = sampled_token[hidden_count:]
                if len(out) > 0:
                    yield out, m.groupdict()
                    hidden_count = 0
                else:
                    hidden_count -= len(sampled_token)

                self.extend_model([sampled_token_ind])



    