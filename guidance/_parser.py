import numpy as np
from ordered_set import OrderedSet
from ._grammar import Join, Select, Terminal, Null, Byte, ByteRange

class EarleyItem:
    __slots__ = ("node", "values", "start", "pos", "log_prob", "children")

    def __init__(self, node, values, pos, start, log_prob):
        if repr(node) == 'hide <- ", 9" | "<s>"\n", 9" <- b\',\' b\' \' b\'9\'\n"<s>" <- b\'<\' b\'s\' b\'>\'\n':
            pass
        self.node = node
        self.values = values
        self.start = start
        self.pos = pos
        self.log_prob = log_prob
        self.children = None

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
        self.state_sets = [OrderedSet()] # the list of Earley 
        self.state_set_pos = 0
        self._add_node(self.grammar, 0, 0.0)
        self._inner_loop(self.state_set_pos)

    @property
    def pos(self):
        return self.state_set_pos
    @pos.setter
    def pos(self, new_pos):
        if new_pos == self.state_set_pos:
            return
        self.state_sets = self.state_sets[:new_pos+1] + [OrderedSet()]
        self.bytes = self.bytes[:new_pos]
        self.state_set_pos = new_pos
        self._inner_loop(self.state_set_pos)

    def _add_node(self, grammar, state_set_pos, log_prob):
        if isinstance(grammar, Terminal):
            new_item = EarleyItem(grammar, tuple(), 0, state_set_pos, log_prob)
            if new_item not in self.state_sets[state_set_pos]:
                self.state_sets[state_set_pos].append(new_item)
            
        elif isinstance(grammar, Join):
            new_item = EarleyItem(grammar, tuple(grammar.values), 0, state_set_pos, log_prob)
            if new_item not in self.state_sets[state_set_pos]:
                self.state_sets[state_set_pos].append(new_item)
        
        elif isinstance(grammar, Select):
            for value in grammar.values:
                new_item = EarleyItem(grammar, (value,), 0, state_set_pos, log_prob)
                if new_item not in self.state_sets[state_set_pos]:
                    self.state_sets[state_set_pos].append(new_item)

    def _inner_loop(self, state_set_pos):
        curr_state_set = self.state_sets[state_set_pos]
        if len(self.state_sets) == state_set_pos + 1:
            self.state_sets.append(OrderedSet())
        next_state_set = self.state_sets[state_set_pos + 1]
        pos = 0
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
                start_state_set = self.state_sets[item.start]
                for start_item in start_state_set:
                    if start_item.pos < len(start_item.values) and start_item.values[start_item.pos] == item.node:
                        curr_state_set.append(EarleyItem(
                            start_item.node,
                            start_item.values,
                            start_item.pos + 1,
                            start_item.start,
                            start_item.log_prob + item.log_prob # increment the log prob by the child value
                        ))
            else:
                # scan
                next_item = item.values[item.pos]
                if isinstance(next_item, Terminal):
                    next_state_set.append(EarleyItem(item.node, item.values, item.pos + 1, item.start, item.log_prob)) # the log prob will get incremented when consume_bytes is called
                
                # prediction
                else:
                    self._add_node(next_item, state_set_pos, 0.0) # the log probs will get incremented by children later

                # handle nullable items by advancing them automatically (since we know we can)
                if next_item.nullable:
                    new_item = EarleyItem(item.node, item.values, item.pos + 1, item.start, item.log_prob)
                    if new_item not in self.state_sets[state_set_pos]:
                        self.state_sets[state_set_pos].append(new_item)
            pos += 1

    def earliest_hidden_start(self):
        '''The earliest that a hidden node might match.
        
        This is useful because it tells us which bytes may end being hidden.
        '''
        earliest_pos = 10000000000
        for item in self.state_sets[self.state_set_pos]:

            if item.pos > 0:
                # check for hidden nodes
                if item.node.hidden and item.start < earliest_pos:
                    earliest_pos = item.start
                
                # check for nodes that are not hidden but end with a hidden terminal (terminal won't be in our state set by themselves, so we need this check)
                else:
                    last_value = item.values[item.pos-1]
                    if isinstance(last_value, Terminal) and last_value.hidden and self.state_set_pos - len(last_value) < earliest_pos:
                        earliest_pos = self.state_set_pos - len(last_value)
        
        return earliest_pos
    
    def matched(self):
        '''Checks if the parser has completely matched the grammar.'''
        for item in self.state_sets[self.state_set_pos]:
            if item.node == self.grammar and item.pos == len(item.values):
                return True
        return False

    def consume_byte(self, byte, log_prob=0.0):
        '''Advances the parser by the given byte.'''
        self.bytes += byte
        next_state_set = self.state_sets[self.state_set_pos + 1]
        new_next_state_set = []
        found_valid = False
        for item in next_state_set:
            if item.pos > 0 and isinstance(item.values[item.pos - 1], Terminal):
                if not item.values[item.pos - 1].match_byte(byte):
                    continue
                else:
                    found_valid = True
            item.log_prob += log_prob # update the probability of the item by the probability of choosing this byte
            new_next_state_set.append(item)
        if not found_valid:
            raise Exception("Attempted to consume a byte that the grammar does not accept!")
        self.state_sets[self.state_set_pos + 1] = OrderedSet(new_next_state_set)
        self.state_set_pos += 1
        self._inner_loop(self.state_set_pos)

    def valid_next_bytes(self):
        '''A list of Byte and ByteRange objects representing the next valid bytes.'''
        valid_items = set()
        next_state_set = self.state_sets[self.state_set_pos + 1]
        for item in next_state_set:
            if item.pos > 0 and isinstance(item.values[item.pos - 1], Terminal):
                v = item.values[item.pos - 1]
                if v not in valid_items:
                    valid_items.add(v)
        return valid_items
    
    def next_byte_mask(self):
        '''A mask version of the `valid_next_bytes` method.'''
        valid_items = self.valid_next_bytes()
        mask = np.zeros(256, dtype=bool)
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
            s += f"\n=== {i} ===\n"
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
                new_state_sets[state.start].append(EarleyItem(state.node, state.values, state.pos, i, state.log_prob))
        
        return new_state_sets
    
    def parse_tree(self):
        reversed_state_sets = self._reversed_state_sets()

        # find the matching root state
        for item in reversed_state_sets[0]:
            if item.node == self.grammar and item.start == len(self.bytes) and item.pos == len(item.values): # note that ".start" mean end because items are reversed
                root_item = item
        self._compute_parse_tree(0, root_item, reversed_state_sets)
        return root_item
    
    def _compute_parse_tree(self, pos, item, reversed_state_sets):
        if pos == 7:
            pass
        
        # compute the children for this item
        assert self._compute_children(pos, item, reversed_state_sets)

        # recurse on the children
        for child in item.children:
            if child is None:
                pass # this child was nullable and was chosen to be null (empty)
            elif isinstance(child, Terminal):
                pos += len(child)
            else:
                self._compute_parse_tree(pos, child, reversed_state_sets)
                pos = child.start # note that ".start" mean end because items are reversed

    def _compute_children(self, state_set_pos, item, reversed_state_sets, values_pos = 0):
            
        # if we are at the end of the values then there no more children and we see if we consumed all the right bytes
        if values_pos == len(item.values):
            return state_set_pos == item.start # note that ".start" mean end because items are reversed
        
        # ensure we have a children array
        if item.children is None:
            item.children = [None for _ in range(len(item.values))]

        # get the child we are trying to match (meaning we are looking for completed early items for this node)
        value = item.values[values_pos]

        # if we have a terminal node we can jump forward that many bytes
        if isinstance(value, Terminal):
            
            # get all states from future children (values)
            if self._compute_children(state_set_pos + len(value), item, reversed_state_sets, values_pos + 1):
                item.children[values_pos] = value
                return True
        
        # otherwise we need to try all possible next matching items in the current state set
        else:
            # loop over every item in the current state set looking for a completed match
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

