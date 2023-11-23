import scipy.special
import scipy.stats
import numpy as np
from .._utils import ByteTrie
from ._model import Model
from .._parser import EarleyCommitParser
from .._grammar import Terminal

class Local(Model):
    def __init__(self, tokens, bos_token_id, eos_token_id=None, echo=True):
        super().__init__(echo)
        
        assert isinstance(tokens[0], bytes), "The tokens need to be provided as bytes!"

        self.tokens = tokens
        self.bos_token_id = bos_token_id
        self.bos_token = None if self.bos_token_id is None else self.tokens[self.bos_token_id]
        self.eos_token_id = eos_token_id if eos_token_id is not None else bos_token_id
        self.eos_token = None if self.eos_token_id is None else self.tokens[self.eos_token_id]

        # build a prefix tree of the tokens
        self._token_trie = ByteTrie(tokens, np.arange(len(tokens)))
        self._token_trie.match = True
        self._token_trie.match_version = 0

        self._max_token_bytes = max([len(t) for t in self.tokens])  

    def _get_logits(self, token_ids, forced_bytes):
        '''A fake method designed to be overriden by subclasses.'''

        # pretend to extend the KV cache and update the log probs
        return np.randn(len(self.tokens))
    
    def _joint_tokenize(self, token_ids):
        # an abstract method. Should return what a full joint tokenizer would give for a given byte string
        return token_ids
        
    def _tokenize_prefix(self, byte_string):
        '''This is used to speed up the tokenization of long prompts without using the parser.'''
        token_ids = []
        token_byte_positions = []
        
        # loop trying to decode a new token at each iteration
        pos = 0
        while True:

            # walk down the token trie looking for a unique token match
            trie = self._token_trie
            valid_pos = -1
            valid_value = -1
            while True:
                if pos >= len(byte_string):
                    if len(trie.children) > 0:
                        valid_pos = -1
                    break

                # check if we can keep going or are at a dead end
                if byte_string[pos:pos+1] in trie.children:
                    trie = trie.children[byte_string[pos:pos+1]]
                    pos += 1

                    # record the last valid token down this path as we go
                    if trie.value is not None:
                        valid_pos = pos
                        valid_value = trie.value
                else:
                    break # we can't go any farther
            
            if valid_pos == -1:
                break
            else:
                token_ids.append(valid_value)
                token_byte_positions.append(valid_pos)
                pos = valid_pos

        return token_ids,token_byte_positions
    
    def _cleanup_tokens(self, token_ids, token_byte_positions):

        # compute a joint tokenization
        joint_token_ids = self._joint_tokenize(token_ids)
        
        # see if we need to redo the tokenization
        redo = False
        if len(joint_token_ids) != len(token_ids):
            redo = True
        else:
            for i,id in enumerate(joint_token_ids):
                if token_ids[i] != id:
                    redo = True
                    break
        
        if redo:
            token_ids = joint_token_ids
            last_pos = token_byte_positions[-1]
            token_byte_positions = []
            pos = 0
            for i,id in enumerate(joint_token_ids):
                pos += len(self.tokens[id])
                token_byte_positions.append(pos)
            assert token_byte_positions[-1] == last_pos
        
        return token_ids, token_byte_positions


    def __call__(self, grammar, max_tokens=100, n=1, top_p=1, temperature=0.0, ensure_bos_token=True, log_probs=False):
        assert n == 1, "Still need to add support for n > 1!"
        
        # get our current context in bytes
        prompt = self._current_prompt()
        prompt = bytes(prompt, encoding="utf8")

        # add the beginning of sequence token if needed
        if ensure_bos_token and self.bos_token is not None and not prompt.startswith(self.bos_token):
            prompt = self.bos_token + prompt
        
        # run a simple tokenizer (that does not use a grammar) on the prefix for better performance
        token_ids,token_byte_positions = self._tokenize_prefix(prompt)
        token_ids,token_byte_positions = self._cleanup_tokens(token_ids,token_byte_positions)
        if len(token_byte_positions) > 0:
            pre_parser_bytes = token_byte_positions[-1]
            prompt = prompt[token_byte_positions[-1]:]
        else:
            pre_parser_bytes = 0
        
        # create a parser with a grammar that includes both our context and the passed grammar
        parser = EarleyCommitParser(prompt + grammar)

        # loop until we have generated a complete pattern
        hidden_count = len(prompt) # we don't emit the prompt
        generated_pos = 0 
        sampled_token_ind = None
        token_count = 0
        last_token_count = 0
        was_forced = False
        while True: # each iteration generates one more token (and some of the associated bytes)

            # enforce the token limit
            if token_count >= max_tokens:
                break

            # note where we are starting for this token
            start_pos = parser.pos

            # let the parser know that we have advanced another token (used ofr tracking max token limits)
            parser.mark_new_token()

            # walk down the trie as far as possible before computing the logits
            retry_token_gen = False
            trie = self._token_trie
            trie.match_version += 1 # this invalidates all the match caches from the previous token
            while True:
                next_byte_mask = parser.next_byte_mask()
                next_byte_mask_sum = next_byte_mask.sum()
                
                # see if we reached a dead end of the grammar
                if next_byte_mask_sum == 0:
                    break
                
                # if there is more than one option we cannot advance without computing the logits 
                elif next_byte_mask_sum != 1:
                    break

                # we are not forced if we are at the end of the grammar
                elif parser.matched():
                    break

                # if there is only one possible next byte we can keep forcing
                elif next_byte_mask_sum == 1:

                    # look for valid children
                    next_byte = None
                    for byte in trie.children:
                        
                        # mark this trie node with an up-to-date match flag (may save work later)
                        node = trie.children[byte]
                        node.match_version = self._token_trie.match_version
                        node.match = next_byte_mask[byte[0]]
                        
                        # see if we found a match
                        if node.match:
                            next_byte = byte
                            break

                    # if we can't extend then this token is forced
                    if next_byte is None:
                        break
                    
                    # otherwise since there is only one possible next byte we keep going
                    else:
                        commit_point = parser.consume_byte(next_byte, log_prob=0.0)
                        
                        # if we are at a hidden commit point then we need to hide the bytes that match that node
                        if commit_point is not None and commit_point.node.hidden:

                            # This takes the item and commits to it as part of the parse and then shrinks it to zero width
                            # in other words this hides the item
                            parser.commit_and_collapse_item(commit_point)
                            
                            # keep the bytes we still need to emit
                            if start_pos < commit_point.start:
                                parser.shadow_rewind(start_pos)
                            
                            else:
                                # pop off any tokens that overlap the hidden bytes
                                i = len(token_byte_positions) - 1
                                while i >= 0 and token_byte_positions[i] - pre_parser_bytes > commit_point.start:
                                    token_ids.pop()
                                    token_byte_positions.pop()
                                    token_count -= 1
                                    i -= 1
                                # re-add any bytes we cut too far on
                                parser.shadow_rewind(token_byte_positions[-1] - pre_parser_bytes)
                            retry_token_gen = True # this restarts us at the top of the outer token gen loop
                            break
                        
                        trie = trie.children[next_byte]
                
            forced_pos = parser.pos # record how far the bytes are forced

            if retry_token_gen:
                continue

            # back up if we got forced up to a point that is not a valid token
            if next_byte_mask_sum <= 1:
                while trie.value is None and trie.parent is not None:
                    trie = trie.parent
                    forced_pos -= 1
                parser.pos = forced_pos
            
            # if we walked all the way to a forced token then we advance without computing the logits
            # we are forced if there are no more options and we are either in the middle of the grammar or at a trie leaf
            is_forced = next_byte_mask_sum <= 1 and (len(trie.children) == 0 if parser.matched() else trie != self._token_trie)
            if is_forced:
                sampled_token_ind = trie.value
                sampled_token = self.tokens[sampled_token_ind]
                new_bytes_log_prob = 0.0
                was_forced = True

            # we are at the end of the grammar
            elif next_byte_mask_sum == 0:
                token_pos = 0

                # mark the token we "sampled" if we have comsumed some bytes
                if trie != self._token_trie:
                    sampled_token_ind = trie.value
                    sampled_token = self.tokens[sampled_token_ind]
                    new_bytes_log_prob = 0.0
                    
            # otherwise we need to compute the logits and sample a valid token
            else:

                # if we were forced we might need to clean up the greedy tokenization to match the global tokenization behavior as seen in training
                if was_forced:
                    token_ids,token_byte_positions = self._cleanup_tokens(token_ids, token_byte_positions)
                    was_forced = False
                logits = self._get_logits(token_ids, parser.bytes[start_pos:forced_pos])

                # if requested we compute the log probabilities so we can track the probabilities of each node
                # TODO: we should lower this step to C++ with pybind11
                if log_probs:
                    _compute_log_probs(trie, scipy.special.log_softmax(logits, dim=-1))

                # get the sampling order
                grammar_temp = parser.next_byte_temperature()
                current_temp = grammar_temp if grammar_temp >= 0 else temperature # we prefer to use the grammar temp when it is specified
                if current_temp == 0:
                    sampling_order = np.argsort(-logits) # we need numpy so the enumerate below does not get really slow...
                else:
                    assert top_p == 1, "Still need to add support for top_p!"
                    probs = scipy.special.softmax(logits / current_temp, axis=-1)
                    sampling_order = np.random.choice(len(probs), size=len(probs), p=probs+1e-10, replace=False) # the 1e-10 is ensure we have no zero probs, which numpy does not like

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
                            
                            # mark that we accepted this byte
                            node = next_node
                            token_pos += 1

                            # get the parser to consume the next byte
                            log_prob_delta = next_node.log_prob - node.log_prob
                            new_bytes_log_prob += log_prob_delta
                            commit_point = parser.consume_byte(next_byte, log_prob=log_prob_delta)
                        
                            # if we are at a hidden commit point then we need to hide the bytes that match that node
                            if commit_point is not None and commit_point.node.hidden:

                                # This takes the item and commits to it as part of the parse and then shrinks it to zero width
                                # in other words this hides the item
                                parser.commit_and_collapse_item(commit_point)
                                
                                # keep the bytes we still need to emit
                                if forced_pos < commit_point.start:
                                    parser.shadow_rewind(forced_pos)
                                
                                else:
                                    # pop off any tokens that overlap the hidden bytes
                                    i = len(token_byte_positions) - 1
                                    while i >= 0 and token_byte_positions[i] - pre_parser_bytes > commit_point.start:
                                        token_ids.pop()
                                        token_byte_positions.pop()
                                        token_count -= 1
                                        i -= 1
                                    # re-add any bytes we cut too far on
                                    parser.shadow_rewind(token_byte_positions[-1] - pre_parser_bytes)
                                retry_token_gen = True # this restarts us at the top of the outer token gen loop
                                break

                            elif token_pos == len(sampled_token):
                                break # this token is valid
                        else:
                            # partially valid tokens are okay if we are running off the end of a grammar, but not otherwise
                            if not parser.matched():
                                token_pos = -1

                            break # this token is no longer valid

                    # see if we are breaking out of the whole loop
                    if retry_token_gen:
                        break

                    # check if this token is dominated by other longer valid tokens (and hence would never be consistent with greedy tokenization)
                    if token_pos == len(sampled_token) and not parser.matched(): # not we don't check if we have matched, because then we can generate anything afterwards
                        if _check_dominated(node, parser, self._token_trie.match_version, parser.next_byte_mask()):
                            token_pos = -1

                    if token_pos > 0:
                        break # we found a valid token

                    if parser.matched():
                        break # if we already have a full match we don't try more tokens we just give up as soon as the model deviates from the grammar
            
            # if we just collpased a hidden commit point then we start over looking for a new token
            if retry_token_gen:
                continue

            # emit whatever we know will not be hidden
            new_bytes = parser.bytes[generated_pos:parser.earliest_hidden_start()]

            # if we cannot consume any more tokens then we are done
            if not is_forced and token_pos < len(sampled_token) and trie == self._token_trie:
                assert parser.matched(), "We can't consume any more tokens, but we are not yet done! Perhaps your model's token set is incomplete?"

                # TODO: if we exactly match the end of the pattern then we can commit to this last token 
                # if m.span()[1] == len(generated_text):
                #     self._cache_state["new_token_ids"].append(sampled_token_ind)

                # capture the named groups from the parse tree
                parse_tree = parser.parse_tree()
                data = {}
                log_prob_data = {}
                _record_captures(parse_tree, data, log_prob_data, parser.bytes)
                
                # we have no valid log prob data if we didn't compute it
                if not log_probs:
                    log_prob_data = {k: None for k in data}
                yield new_bytes[hidden_count:], not is_forced, new_bytes_log_prob, data, log_prob_data, token_count - last_token_count
                last_token_count = token_count
                break # we are done!
            else:
                generated_pos += len(new_bytes)

                # yeild the snippet of text created by the next token
                out = new_bytes[hidden_count:]
                if len(out) > 0:
                    yield out, not is_forced, new_bytes_log_prob, {}, {}, token_count - last_token_count # note that we don't capture groups until a complete parse right now...
                    last_token_count = token_count
                    hidden_count = 0
                    token_count += 1 # note we only update this for tokens that emit non-hidden content
                else:
                    hidden_count -= len(new_bytes)

                token_ids.append(sampled_token_ind)

                # track the byte position of each token
                if len(token_byte_positions) == 0:
                    token_byte_positions.append(len(sampled_token))
                else:
                    token_byte_positions.append(token_byte_positions[-1] + len(sampled_token))

def _record_captures(initial_item, data, log_prob_data, byte_data):
    stack = [(initial_item, 0)]
    used_names = set() # track which capture names have been used so self-recursive children don't overwrite their parents
    
    while stack:
        item, byte_pos = stack.pop()
        # terminal nodes
        if isinstance(item, Terminal):

            # if we are at a capture group node then we save the matched terminal byte
            if item.capture_name is not None:
                data[item.capture_name] = item.byte
                log_prob_data[item.capture_name] = 0
        
        # internal nodes
        else:
            start_byte_pos = byte_pos

            # recurse for all our non-null children
            for child in item.children:
                if child is not None:
                    stack.append((child, byte_pos))
                    # _record_captures(child, data, log_prob_data, byte_data, byte_pos)
                    if isinstance(child, Terminal):
                        byte_pos += len(child)
                    else:
                        byte_pos = child.start # note that "start" means "end" since this is a reversed state set

            # if we are at a capture group node then we save the matched bytes range
            # note that we record this after calling our children so that we save the outermost version of self-recursive calls
            cname = item.node.capture_name
            if cname is not None and cname not in used_names:
                
                # see if we are doing a list append
                if cname.startswith("__LIST_APPEND:"):
                    cname = cname[14:] # trim off the list append tag
                    if cname not in data or not isinstance(data[cname], list):
                        data[cname] = []
                        log_prob_data[cname] = []
                    data[cname].append(byte_data[start_byte_pos:item.start])
                    log_prob_data[cname].append(item.log_prob)
                
                # or just a regular assignment
                else:
                    data[cname] = byte_data[start_byte_pos:item.start] # note that "start" means "end" since this is a reversed state set
                    log_prob_data[cname] = item.log_prob

                used_names.add(cname)    

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
