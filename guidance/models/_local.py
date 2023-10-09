import torch
import numpy as np
import numpy as np
from .._utils import ByteTrie
from ._model import Model
# from ..library._string import string
from .._parser import EarleyCommitParser

class Local(Model):
    def __init__(self, tokens, bos_token_id, eos_token_id=None, echo=True):
        super().__init__(echo)
        
        assert isinstance(tokens[0], bytes), "The tokens need to be provided as bytes!"

        self.tokens = tokens
        self.bos_token_id = bos_token_id
        self.bos_token = self.tokens[self.bos_token_id]
        self.eos_token_id = eos_token_id if eos_token_id is not None else bos_token_id
        self.eos_token = self.tokens[self.eos_token_id]

        # build a prefix tree of the tokens
        self._token_trie = ByteTrie(tokens, np.arange(len(tokens)))
        self._token_trie.match = True
        self._token_trie.match_version = 0

        # all mutable state goes in the cache dictionary
        self._cache_state["cache_token_ids"] = []
        self._cache_state["new_token_ids"] = []

    def _get_logits(self):
        '''A fake method designed to be overriden by subclasses.'''
        self._cache_state["cache_token_ids"].extend(self._new_token_ids)
        self._new_token_ids = []

        # pretend to extend the KV cache and update the log probs
        return torch.randn(len(self.tokens))

    def _longest_token_match(self, bytes):
        '''Greedy token matching.'''
        # if string.startswith("\n"):
        #     pass
        trie_pos = self._token_trie
        for i,c in enumerate(bytes):
            if c in trie_pos.children:
                trie_pos = trie_pos.children[c]
            else:
                return bytes[:i], trie_pos.value # note that if there are redudant tokens we choose the one stored in the trie
        if len(trie_pos.children) == 0:
            return bytes[:i+1], trie_pos.value
        else:
            return None,None # more than one token can match these bytes

    def __call__(self, grammar, max_tokens=100, n=1, top_p=1, temperature=0.0, ensure_bos_token=True):
        assert n == 1, "Still need to add support for n > 1!"
        prompt = str(self)
        prompt = bytes(prompt, encoding="utf8")

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
            prompt = prompt[cache_pos:]
        self._cache_state["new_token_ids"] = []
        
        # move whatever is not cached from the prompt into the grammar (since the grammar is what we will generate)
        # note we also anchor the pattern to the start of the sequence
        grammar = prompt + grammar
        parser = EarleyCommitParser(grammar)

        # loop until we have generated a complete pattern
        # call_count = [0]
        hidden_count = len(prompt) # we don't emit the prompt
        generated_pos = 0 
        sampled_token_ind = None
        for token_count in range(max_tokens):

            # note where we are starting
            start_pos = parser.pos

            # walk down the trie as far as possible before computing the logits
            self._token_trie.match_version += 1 # this invalidates all the match caches from the previous token
            trie = self._token_trie
            while True:
                next_byte_mask = parser.next_byte_mask()

                # look for valid children
                found = None
                found_many = False
                for byte in trie.children:
                    
                    # mark this trie node with an up-to-date match flag (may save work later)
                    node = trie.children[byte]
                    node.match_version = self._token_trie.match_version
                    node.match = next_byte_mask[byte[0]]
                    
                    # track if we have one or more than one match
                    if node.match:
                        if found is not None:
                            found_many = True
                        else:
                            found = byte

                # if there are none or several possible next bytes we can't walk farther without logits
                if found_many or found is None:
                    break 
                
                # otherwise there is only one possible next byte so we keep going
                else:
                    parser.consume_byte(found)
                    trie = trie.children[found]
            forced_pos = parser.pos # record how far the bytes are forced
                    
            # if we walked all the way to a forced token then we advance without computing the logits
            if found is None:
                sampled_token_ind = trie.value
                sampled_token = self.tokens[sampled_token_ind]

            # otherwise we need to compute the logits and sample a valid token
            else:
                logits = self._get_logits()

                # get the sampling order
                if temperature == 0:
                    sampling_order = torch.argsort(logits, descending=True).cpu().numpy() # we need numpy so the enumerate below does not get really slow...
                else:
                    assert top_p == 1, "Still need to add support for top_p!"
                    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                    sampling_order = torch.multinomial(probs, len(probs)).cpu().numpy()

                # loop over the tokens looking for a valid one
                for i,sampled_token_ind in enumerate(sampling_order):
                    sampled_token = self.tokens[sampled_token_ind]

                    # make sure the parse is backed up to the position we want to start checking from TODO: make this account for shared prefixes with the last token
                    parser.pos = forced_pos

                    # make sure it matches any forced prefix
                    if start_pos < forced_pos and not sampled_token.startswith(parser.bytes[start_pos:forced_pos]):
                        continue
                    offset = forced_pos - start_pos

                    # check to see if the sampled token is allowed
                    token_pos = offset
                    node = trie # this is the Trie node we were left at when we could force the next byte above

                    while token_pos < len(sampled_token):
                        next_byte = sampled_token[token_pos:token_pos+1]
                        next_node = node.children[next_byte]

                        # if we don't have a cached match flag compute it using the grammar
                        if next_node.match_version < self._token_trie.match_version:
                            next_byte_mask = parser.next_byte_mask()
                            for byte in node.children: # we update all the children since the parser knows the full mask
                                child = node.children[byte]
                                child.match_version = self._token_trie.match_version
                                child.match = next_byte_mask[byte[0]]
                        
                        # advance or fail according to the (now up-to-date) match cache
                        if next_node.match:
                            node = next_node
                            parser.consume_byte(next_byte)
                            token_pos += 1
                            if token_pos == len(sampled_token) or parser.matched():
                                break # this token is valid
                        else:
                            token_pos = -1
                            break # this token is invalid

                    if token_pos > 0:
                        break # we found a valid token
            
            # emit whatever we know will not be hidden
            new_bytes = parser.bytes[generated_pos:parser.earliest_hidden_start()]

            # if we have a full match we are done
            if parser.matched():

                # TODO: if we exactly match the end of the pattern then we can commit to this last token 
                # if m.span()[1] == len(generated_text):
                #     self._cache_state["new_token_ids"].append(sampled_token_ind)
                
                if hidden_count < len(new_bytes):
                    yield new_bytes[hidden_count:len(new_bytes)], {} # TODO also return captured ranges once the parser supports that
                break # we are done!
            else:
                generated_pos += len(new_bytes)

                # yeild the snippet of text created by the next token
                out = new_bytes[hidden_count:]
                if len(out) > 0:
                    yield out, {} # TODO also return captured ranges once the parser supports that
                    hidden_count = 0
                else:
                    hidden_count -= len(new_bytes)

                self._cache_state["new_token_ids"].append(sampled_token_ind)