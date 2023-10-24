try:
    import torch
except ImportError:
    pass
import numpy as np
from .._utils import ByteTrie
from ._model import Model
# from ..library._string import string
from .._parser import EarleyCommitParser
# import numba

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

    def __call__(self, grammar, max_tokens=100, n=1, top_p=1, temperature=0.0, ensure_bos_token=True, log_probs=False):
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
        token_count = 0
        while True:

            # enforce the token limit
            if token_count >= max_tokens:
                break

            # note where we are starting
            start_pos = parser.pos

            # walk down the trie as far as possible before computing the logits
            self._token_trie.match_version += 1 # this invalidates all the match caches from the previous token
            trie = self._token_trie
            found = None
            while True:

                # see if we completed the grammar
                if parser.matched():
                    found = None
                    break

                next_byte_mask = parser.next_byte_mask()
                next_byte_mask_sum = next_byte_mask.sum()
                
                # see if we reached a dead end of the grammar
                if next_byte_mask_sum == 0:
                    found = None
                    break

                # we can advance without logits when there is only one option
                elif next_byte_mask_sum != 1:
                    break

                # look for valid children
                found = None
                for byte in trie.children:
                    
                    # mark this trie node with an up-to-date match flag (may save work later)
                    node = trie.children[byte]
                    node.match_version = self._token_trie.match_version
                    node.match = next_byte_mask[byte[0]]
                    
                    # see if we found a match
                    if node.match:
                        found = byte
                        break

                # if we can't extend then this token is forced
                if found is None:
                    break
                
                # otherwise since there is only one possible next byte we keep going
                else:
                    parser.consume_byte(found, log_prob=0.0)
                    trie = trie.children[found]
            forced_pos = parser.pos # record how far the bytes are forced

            # back up if we got stuck at a point that is not a valid token
            if found is None:
                while trie.value is None and trie.parent is not None:
                    trie = trie.parent
                    forced_pos -= 1
                parser.pos = forced_pos
            
            # if we walked all the way to a forced token then we advance without computing the logits
            is_forced = found is None and trie != self._token_trie
            if is_forced:
                sampled_token_ind = trie.value
                sampled_token = self.tokens[sampled_token_ind]
                new_bytes_log_prob = 0.0

            # otherwise we need to compute the logits and sample a valid token
            elif not parser.matched():
                logits = self._get_logits()

                # if requested we compute the log probabilities so we can track the probabilities of each node
                # TODO: we should lower this step to C++ or use numba because it is quite slow
                if log_probs:
                    _compute_log_probs(trie, torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy())

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
                    new_bytes_log_prob = 0.0

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
                            log_prob_delta = next_node.log_prob - node.log_prob
                            new_bytes_log_prob += log_prob_delta
                            parser.consume_byte(next_byte, log_prob=log_prob_delta)
                            node = next_node
                            token_pos += 1
                            if token_pos == len(sampled_token) or parser.matched():
                                break # this token is valid
                        else:
                            token_pos = -1
                            break # this token is invalid

                    # check if this token is dominated by other longer valid tokens (and hence would never be consistent with greedy tokenization)
                    if token_pos == len(sampled_token): 
                        if _check_dominated(node, parser, self._token_trie.match_version, parser.next_byte_mask()):
                            token_pos = -1

                    if token_pos > 0:
                        break # we found a valid token
            
            # emit whatever we know will not be hidden
            new_bytes = parser.bytes[generated_pos:parser.earliest_hidden_start()]

            # if we have a full match we are done
            if parser.matched():

                # TODO: if we exactly match the end of the pattern then we can commit to this last token 
                # if m.span()[1] == len(generated_text):
                #     self._cache_state["new_token_ids"].append(sampled_token_ind)

                reversed_state_sets = parser._reversed_state_sets()

                # find the matching root state
                for state in reversed_state_sets[0]:
                    if state.node == parser.grammar and state.start == len(parser.bytes) and state.pos == len(state.values):
                        root_state = state
                
                # value_states = _parsed_value_states(root_state, 0, 0, reversed_state_sets)
                data = {}
                log_prob_data = {}
                _record_names(root_state, 0, reversed_state_sets, data, log_prob_data, parser.bytes)
                
                # we have no valid log prob data if we didn't compute it
                if not log_probs:
                    log_prob_data = {k: None for k in data}

                if hidden_count < len(new_bytes):
                    yield new_bytes[hidden_count:len(new_bytes)], not is_forced, new_bytes_log_prob, data, log_prob_data
                break # we are done!
            else:
                generated_pos += len(new_bytes)

                # yeild the snippet of text created by the next token
                out = new_bytes[hidden_count:]
                if len(out) > 0:
                    yield out, not is_forced, new_bytes_log_prob, {}, {} # note that we don't capture groups until a complete parse right now...
                    hidden_count = 0
                    token_count += 1 # note we only update this for tokens that emit non-hidden content
                else:
                    hidden_count -= len(new_bytes)

                self._cache_state["new_token_ids"].append(sampled_token_ind)

def _record_names(state, state_pos, reversed_state_sets, data, log_probs, byte_data):
    '''Extract all the named capture groups from the parser.'''
    
    # if we are at a capture group node then we save the matched bytes range
    if state.node.capture_name is not None:
        data[state.node.capture_name] = byte_data[state_pos:state.start] # note that "start" means "end" since this is a reversed state set
        log_probs[state.node.capture_name] = state.log_prob
    
    # get all the completed state corresponding to this state's node's children (values)
    value_states = _parsed_value_states(state, state_pos, reversed_state_sets)

    # for each such completed state we recursively look for capture groups
    if value_states is not None:
        for value_state in value_states:
            _record_names(value_state, state_pos, reversed_state_sets, data, log_probs, byte_data)
            state_pos = value_state.start # note that "start" means "end" since this is a reversed state set

def _parsed_value_states(state, state_pos, reversed_state_sets, values_pos = 0):
    '''Get the completed states (reversed earley items) of all the children nodes.'''

    # if we are at the end of the values then there no more children
    if values_pos == len(state.values):
        return []

    # get the child we are trying to match (meaning we are looking for completed early items for this node)
    value = state.values[values_pos]

    # loop over every item in the current state set looking for a match
    for inner_state in reversed_state_sets[state_pos]:
        if inner_state.node == value and inner_state.pos == len(inner_state.values):

            # get all states from future children (values)
            value_states = _parsed_value_states(state, inner_state.start, reversed_state_sets, values_pos + 1)
            
            # we break out once we get our first match
            if value_states is not None:
                return [inner_state] + value_states
    
    return None
        

def _compute_log_probs(trie, log_probs):
    '''Computes the log probabilities for each internal trie node.'''
    if trie.value is not None:
        trie.log_prob += log_probs[trie.value]
    
    if len(trie.children) > 0:
        child_log_probs = []
        for b in trie.children:
            child = trie.children[b]
            _compute_log_probs(child, log_probs)
            child_log_probs.append(child.log_prob)
        trie.log_prob = np.logaddexp.reduce(child_log_probs)

def _check_dominated(node, parser, match_version, next_byte_mask):
    curr_pos = parser.pos
    for byte_num in next_byte_mask.nonzero()[0]:
        next_byte = bytes((byte_num,))
        if next_byte not in node.children:
            return False # no possible exension this direction, so we are not dominated
        child = node.children[next_byte]
        if child.match_version < match_version:
            child.match_version = match_version
            child.match = next_byte_mask[next_byte[0]]
        
        if not child.match:
            return False # this child does not dominate the node, so the node is not dominated
        elif child.value is None: # this child might not dominate the node
            parser.consume_byte(next_byte, log_prob=0.0)
            child_dominate = _check_dominated(child, parser, match_version, parser.next_byte_mask())
            parser.pos = curr_pos
            if not child_dominate:
                return False
    return True
            

# this token can only be dominated if we are at the end of this token

# dominated = True # assume true until proven otherwise
# check_stack = [(node, parser.pos)]
# for byte_num in next_byte_mask.nonzero()[0]:
#     node, pos = check_stack.pop()
#     if node.match_version < self._token_trie.match_version:
#         parser.pos
#         next_byte_mask = parser.next_byte_mask()
#         for byte in node.children: # we update all the children since the parser knows the full mask
#             child = node.children[byte]
#             child.match_version = self._token_trie.match_version
#             child.match = next_byte_mask[byte[0]]
#     byte = bytes((byte_num,))
#     child = node.children[byte]
#     if not child.match:
#         dominated = False
#         break
#     else:
#         # if this is not a valid token yet we need to determine if this child is dominated
#         if child.value is None:
#             check_stack.push((child, pos))

# we invalidate this token if it is dominated
# if dominated:
#     token_pos = -1