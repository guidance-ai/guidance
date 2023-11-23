import numpy as np
from ordered_set import OrderedSet
from ._grammar import Join, Select, Terminal, Null, Byte, ByteRange

class EarleyItem:
    __slots__ = ("node", "values", "start", "pos", "log_prob", "children", "hidden_start")

    def __init__(self, node, values, pos, start, log_prob, hidden_start):
        self.node = node
        self.values = values
        self.start = start
        self.pos = pos
        self.log_prob = log_prob
        self.children = None
        self.hidden_start = hidden_start

    def __eq__(self, other):
        return isinstance(other, EarleyItem) and \
               self.start == other.start and \
               self.pos == other.pos and \
               self.node == other.node and \
               self.values == other.values and \
               self.log_prob == other.log_prob
    
    def __hash__(self):
        return hash((self.node, self.values, self.start, self.pos))
    
    def __repr__(self):
        if isinstance(self.node, Join):
            s = f"{self.node.name:20} -> "
            rs = ""
            for i,v in enumerate(self.values):
                if self.pos == i:
                    rs += "•"
                rs += v.name + " "
            if self.pos == len(self.values):
                rs += "•"
        elif isinstance(self.node, Select):
            s = f"{self.node.name:20} -> "
            rs = ""
            if self.pos == 0:
                rs += "•"
            rs += self.values[0].name
            if self.pos == 1:
                rs += "•"
        else:
            assert False
        return s + f"{rs:40} ({self.start}) {'nullable' if self.node.nullable else ''}"

class EarleyCommitParser:
    def __init__(self, grammar):

        # we can't have a terminal as the root
        if isinstance(grammar, Terminal):
            grammar = Join([grammar])
        
        self.grammar = grammar
        self.bytes = b''
        self.state_sets = [OrderedSet()] # the list of Earley items for each byte
        self.token_counts = [] # used to track how many tokens have been used
        self.state_set_pos = 0
        self.shadow_pos = 0
        self._add_node(self.grammar, 0, 0.0, 1000000000)
        self._inner_loop(self.state_set_pos)

    @property
    def pos(self):
        return self.shadow_pos
    @pos.setter
    def pos(self, new_pos):

        # do nothing if we aren't moving
        if new_pos == self.state_set_pos:
            return
        elif new_pos > self.state_set_pos:
            raise Exception("Can't move the parser position forward! (only backward)")
        
        # check if we are just moving the shadow position
        if new_pos >= self.shadow_pos:
            self.shadow_pos = new_pos
            return
        
        # actually reset our position if we need to
        self.state_sets = self.state_sets[:new_pos+1] + [OrderedSet()]
        self.token_counts = self.token_counts[:new_pos+2]
        self.bytes = self.bytes[:new_pos]
        self.state_set_pos = new_pos
        self.shadow_pos = new_pos
        self._inner_loop(self.state_set_pos)

    def _add_item(self, state_set_pos, new_item):
        state_set = self.state_sets[state_set_pos]
        if new_item not in state_set:
            state_set.append(new_item)
        else:
            existing_item = state_set.items[state_set.map[new_item]]
            existing_item.hidden_start = min(existing_item.hidden_start, new_item.hidden_start)

    def _add_node(self, grammar, state_set_pos, log_prob, hidden_start):
        if isinstance(grammar, Terminal):
            new_item = EarleyItem(grammar, tuple(), 0, state_set_pos, log_prob, hidden_start)
            self._add_item(state_set_pos, new_item)
            
        elif isinstance(grammar, Join):
            new_item = EarleyItem(grammar, tuple(grammar.values), 0, state_set_pos, log_prob, hidden_start)
            self._add_item(state_set_pos, new_item)
        
        elif isinstance(grammar, Select):
            for value in grammar.values:
                new_item = EarleyItem(grammar, (value,), 0, state_set_pos, log_prob, hidden_start)
                self._add_item(state_set_pos, new_item)

    def _inner_loop(self, state_set_pos, start_pos=0):
        curr_state_set = self.state_sets[state_set_pos]
        if len(self.state_sets) == state_set_pos + 1:
            self.state_sets.append(OrderedSet())
            self.token_counts.append(self.token_counts[-1] if len(self.token_counts) > 0 else 0)
        next_state_set = self.state_sets[state_set_pos + 1]
        pos = start_pos
        while len(curr_state_set) > pos:
            item = curr_state_set[pos]

            # completion
            if item.pos == len(item.values):
                
                # if we complete an item that is a "commit point" then we eliminate all other possible
                # parses so that we are "committed" to using this item
                # we do this by removing any unprocessed items in the current state set and clearing the next state set
                if item.node.commit_point:
                    while len(curr_state_set) > pos + 1:
                        curr_state_set.pop()
                    next_state_set.clear()
                
                # advance all the parents that our completion impacts
                token_span = self.token_counts[state_set_pos] - self.token_counts[item.start]
                start_state_set = self.state_sets[item.start]
                for start_item in start_state_set:
                    if start_item.pos < len(start_item.values) and start_item.values[start_item.pos] == item.node:
                        
                        if item.node.max_tokens <= token_span and any(start_item.node == v  for v in item.node.values):
                            continue # skip advancing parents that are also children (recursion) once we are past the token limit

                        curr_state_set.append(EarleyItem(
                            start_item.node,
                            start_item.values,
                            start_item.pos + 1,
                            start_item.start,
                            start_item.log_prob + item.log_prob, # increment the log prob by the child value,
                            start_item.hidden_start
                        ))
            
            # don't advance past our max token limit
            elif item.node.max_tokens > self.token_counts[state_set_pos] - self.token_counts[item.start]:

                # scan (note we only scan forward when we have more max token headroom left)
                next_item_node = item.values[item.pos]
                hidden_start = item.hidden_start
                if next_item_node.hidden:
                    hidden_start = min(state_set_pos, hidden_start)
                if isinstance(next_item_node, Terminal):# and item.node.max_tokens > self.token_counts[state_set_pos] - self.token_counts[item.start]:
                    next_state_set.append(EarleyItem(item.node, item.values, item.pos + 1, item.start, item.log_prob, hidden_start)) # the log prob will get incremented when consume_bytes is called
                
                # prediction
                else:
                    self._add_node(next_item_node, state_set_pos, 0.0, hidden_start) # the log probs will get incremented by children later

                # handle nullable items by advancing them automatically (since we know we can)
                if next_item_node.nullable:
                    new_item = EarleyItem(item.node, item.values, item.pos + 1, item.start, item.log_prob, item.hidden_start)
                    if new_item not in self.state_sets[state_set_pos]:
                        self.state_sets[state_set_pos].append(new_item)
            pos += 1

    def earliest_hidden_start(self, state_pos=None):
        '''The earliest that a hidden node might match.
        
        This is useful because it tells us which bytes may end being hidden.
        '''
        if state_pos is None:
            state_pos = self.state_set_pos
        earliest_pos = 10000000000
        for item in self.state_sets[state_pos]:
            earliest_pos = min(earliest_pos, item.hidden_start)
        return earliest_pos
    
    # def earliest_hidden_start(self, state_pos=None):
    #     '''The earliest that a hidden node might match.
        
    #     This is useful because it tells us which bytes may end being hidden.
    #     '''
    #     if state_pos is None:
    #         state_pos = self.state_set_pos
    #     earliest_pos = 10000000000
    #     for item in self.state_sets[state_pos]:

    #         if item.pos > 0:
    #             # check for hidden nodes
    #             if item.node.hidden and item.start < earliest_pos:
    #                 earliest_pos = item.start
                
    #             # check for nodes that are not hidden but end with a hidden terminal (terminal won't be in our state set by themselves, so we need this check)
    #             else:
    #                 last_value = item.values[item.pos-1]
    #                 if isinstance(last_value, Terminal) and last_value.hidden and state_pos - len(last_value) < earliest_pos:
    #                     earliest_pos = state_pos - len(last_value)
        
    #     return earliest_pos
    
    def matched(self):
        '''Checks if the parser has completely matched the grammar.'''
        if self.shadow_pos != self.state_set_pos:
            return False
        for item in self.state_sets[self.state_set_pos]:
            if item.node == self.grammar and item.pos == len(item.values):
                return True
        return False
    
    def shadow_rewind(self, new_pos):
        if new_pos == self.state_set_pos:
            return
        self.shadow_pos = new_pos
    
    def commit_and_collapse_item(self, item):
        '''This collapses the item into zero width and rewinds the parser position accordingly.
        
        Note we assume the item is in the current state set.
        '''

        # trim off the state sets that matches this item
        self.state_sets = self.state_sets[:item.start + 1]
        self.token_counts = self.token_counts[:item.start + 1]
        self.bytes = self.bytes[:item.start]
        self.state_set_pos = item.start
        self.shadow_pos = item.start

        # add this state to its start point (making it a zero length match with no values)
        self.state_sets[item.start].append(EarleyItem(item.node, tuple(), 0, item.start, item.log_prob, item.hidden_start))

        # expand from this state
        self._inner_loop(item.start, len(self.state_sets[item.start]) - 1)

    def mark_new_token(self):
        # TODO: we allow ourselves to go one past our max token limit when we hit a one-byte token
        #       because we don't know if we are continuing or extending a new token when we parse
        #       the first byte of the token. We could fix this by rerunning the inner_loop after each
        #       token, but we skip that for now since max_tokens is not a hard garuntee anyway when you
        #       have patterns.
        
        self.token_counts[-1] += 1

    def consume_byte(self, byte, log_prob=0.0):
        '''Advances the parser by the given byte.'''

        # see if we need to advance our shadow position...
        if self.shadow_pos < self.state_set_pos:
            assert byte == self.bytes[self.shadow_pos:self.shadow_pos+1], "Attempted to consume a byte by advancing shadow_pos but the byte didn't match!"
            self.shadow_pos += 1
            return

        # ...if not, we extend our bytes
        self.bytes += byte

        # filter out all the extensions that don't match this byte
        new_next_state_set = []
        found_valid = False
        found_invalid = False
        hidden_start = 10000000000
        for item in self.state_sets[self.state_set_pos + 1]:
            if item.pos > 0 and isinstance(item.values[item.pos - 1], Terminal):
                last_inner_node = item.values[item.pos - 1]
                if not last_inner_node.match_byte(byte):
                    found_invalid = True
                    continue
                else:
                    found_valid = True
                    if last_inner_node.commit_point:
                        item.log_prob += log_prob
                        new_next_state_set = [item]
                        hidden_start = min(hidden_start, item.hidden_start)
                        found_invalid = True # we make everything else invalid, so that means we found something invalid
                        break
            item.log_prob += log_prob # update the probability of the item by the probability of choosing this byte
            new_next_state_set.append(item)
            hidden_start = min(hidden_start, item.hidden_start)
        if not found_valid:
            raise Exception("Attempted to consume a byte that the grammar does not accept!")
        if found_invalid: # only update if we changed the set
            self.state_sets[self.state_set_pos + 1] = OrderedSet(new_next_state_set)

        # advance the parser one position
        self.state_set_pos += 1
        self.shadow_pos += 1
        self._inner_loop(self.state_set_pos)

        # look for a commit point node
        commit_point = None
        for item in self.state_sets[self.state_set_pos]:
            if item.node.commit_point and item.pos == len(item.values) or (item.pos > 0 and item.values[item.pos-1].commit_point):
                commit_point = item
                break # TODO: consider how we might need to prioritize multiple commit point nodes (an uncommon scenario I think)
        # hidden_start, 
        return commit_point

    def valid_next_bytes(self):
        '''A list of Byte and ByteRange objects representing the next valid bytes.'''
        valid_items = set()
        next_state_set = self.state_sets[self.state_set_pos + 1]
        for item in next_state_set:
            token_span = self.token_counts[-1] - self.token_counts[item.start]
            if item.node.max_tokens <= token_span:
                continue
            elif item.pos > 0 and isinstance(item.values[item.pos - 1], Terminal):
                v = item.values[item.pos - 1]
                if v not in valid_items:
                    valid_items.add(v)
        return valid_items
    
    def next_byte_temperature(self):
        '''The maximum temperature over all the next bytes, or -1 if no temperature is set.'''
        max_temp = -1
        next_state_set = self.state_sets[self.state_set_pos + 1]
        for item in next_state_set:
            if item.pos > 0 and isinstance(item.values[item.pos - 1], Terminal):
                v = item.values[item.pos - 1]
                max_temp = max(max_temp, v.temperature)
        return max_temp
    
    def next_byte_mask(self):
        '''A mask version of the `valid_next_bytes` method.'''
        
        mask = np.zeros(256, dtype=bool)

        # if we are shadow rewound then we just force those bytes again
        if self.shadow_pos < self.state_set_pos:
            mask[self.bytes[self.shadow_pos]] = True
        
        # otherwise we compute the valid bytes from the grammar
        else:
            valid_items = self.valid_next_bytes()
            for item in valid_items:
                if isinstance(item, Byte):
                    mask[item.byte[0]] = True
                elif isinstance(item, ByteRange):
                    mask[item.byte_range[0]:item.byte_range[1]+1] = True
                else:
                    raise Exception("Unknown Terminal Type: "  + str(type(item)))
        return mask

    def __repr__(self, state_sets=None) -> str:
        s = ""
        if state_sets is None:
            state_sets = self.state_sets
        for i,states in enumerate(state_sets):
            s += f"\n=== {i} ==="
            if self.state_set_pos == i:
                s += " (state_set_pos)"
            s += "\n"
            for state in states:
                if isinstance(state.node, Join):
                    s += f"{state.node.name:20} -> "
                    rs = ""
                    for i,v in enumerate(state.values):
                        if state.pos == i:
                            rs += "•"
                        rs += v.name + " "
                    if state.pos == len(state.values):
                        rs += "•"
                elif isinstance(state.node, Select):
                    s += f"{state.node.name:20} -> "
                    rs = ""
                    if state.pos == 0:
                       rs += "•"
                    if len(state.values) == 0:
                        rs += "NO_VALUES!"
                    else:
                        rs += state.values[0].name
                        if state.pos == 1:
                            rs += "•"
                else:
                    assert False
                s += f"{rs:40} ({state.start}) {'nullable' if state.node.nullable else ''}\n"
        return s
    
    def _reversed_state_sets(self):
        new_state_sets = [OrderedSet([]) for _ in range(len(self.state_sets))]
        for i,states in enumerate(self.state_sets):
            for state in states:
                # if state.node.name == "__call___c":
                #     pass
                new_state_sets[state.start].append(EarleyItem(state.node, state.values, state.pos, i, state.log_prob, state.hidden_start))
        
        return new_state_sets
    
    def parse_tree(self):
        reversed_state_sets = self._reversed_state_sets()

        # find the matching root state
        for item in reversed_state_sets[0]:
            if item.node == self.grammar and item.start == len(self.bytes) and item.pos == len(item.values): # note that ".start" mean end because items are reversed
                root_item = item
        self._compute_parse_tree(0, root_item, reversed_state_sets)
        return root_item
    
    def _compute_parse_tree(self, initial_pos, initial_item, reversed_state_sets):
        stack = [(initial_pos, initial_item)]
        
        while stack:
            pos, item = stack.pop()

            # compute the children for this item
            assert self._compute_children(pos, item, reversed_state_sets)

            # recurse on the children
            for child in item.children:
                if child is None:
                    pass # this child was nullable and was chosen to be null (empty)
                elif isinstance(child, Terminal):
                    pos += len(child)
                else:
                    stack.append((pos, child))
                    pos = child.start # note that ".start" mean end because items are reversed

    def _compute_children(self, state_set_pos, item, reversed_state_sets, values_pos = 0):

        # ensure we have a children array
        if item.children is None:
            item.children = [None for _ in range(len(item.values))]

        # consume as many terminal children as possible
        while True:
            
            # if we are at the end of the values then there no more children and we see if we consumed all the right bytes
            if values_pos == len(item.values):
                return state_set_pos == item.start # note that ".start" mean end because items are reversed

            # get the child we are trying to match (meaning we are looking for completed early items for this node)
            value = item.values[values_pos]

            # if we have a terminal node we can jump forward that many bytes
            if isinstance(value, Terminal):
                item.children[values_pos] = value
                values_pos += 1
                state_set_pos += len(value)
            else:
                break
            
        # otherwise we need to try all possible next matching items in the current state set
        # so we loop over every item in the current state set looking for a completed match
        for inner_item in reversed_state_sets[state_set_pos]:
            if inner_item.node == value and inner_item.pos == len(inner_item.values):

                # see if we can get a complete parse following this inner item
                if self._compute_children(inner_item.start, item, reversed_state_sets, values_pos + 1):
                    item.children[values_pos] = inner_item
                    return True
                    
        # if we didn't find a child set and this is nullable we can skip this child (since it may not exist if nulled)
        if value.nullable:
            if self._compute_children(state_set_pos, item, reversed_state_sets, values_pos + 1):
                item.children[values_pos] = None # this child was skipped since it was nullable
                return True
        
        return False

