import guidance
# from ._prefix_tree import prefix_tree
from .._grammar import string, select
from ._optional import optional


# @guidance(stateless=True, dedent=False)
# def substring(lm, s):
    
#     # build a prefix tree over all suffixes
#     strings = [s[i:] for i in range(len(s))]
#     return lm + prefix_tree(strings, partial_matches=True)

@guidance(stateless=True, dedent=False)
def substring_simple(lm, s):
    grammars = ['' for _ in range(len(s))]
    grammars[-1] = select(['', string(s[-1])])
    for i in range(len(s) - 2, -1, -1):
        si = string(s[i])
        grammars[i] = select([si, si + grammars[i + 1]])
    return lm + select(grammars)

class State:
    def __init__(self):
        self.len = 0
        self.link = -1
        self.next = {}

class SuffixAutomaton:
    '''A suffix automoton.

    For details see https://en.wikipedia.org/wiki/Suffix_automaton.
    
    Implementation is based on https://cp-algorithms.com/string/suffix-automaton.html
    '''
    def __init__(self, s):
        
        # init our variables
        self.MAXLEN = 2*len(s) # we can't have more than 2n-1 states for a string of length n
        self.states = [State() for _ in range(self.MAXLEN * 2)]
        self.states[0].len = 0
        self.states[0].link = -1
        self.sz = 1 # init with a single state
        self.last = 0

        # Add characters to the suffix automaton
        for character in s:
            self.sa_extend(character)

    @property
    def root(self):
        return self.states[0]
    
    def sa_extend(self, c):
        cur = self.sz
        self.sz += 1
        self.states[cur].len = self.states[self.last].len + 1

        p = self.last
        while p != -1 and c not in self.states[p].next:
            self.states[p].next[c] = cur
            p = self.states[p].link

        if p == -1:
            self.states[cur].link = 0
        else:
            q = self.states[p].next[c]
            if self.states[p].len + 1 == self.states[q].len:
                self.states[cur].link = q
            else:
                clone = self.sz
                self.sz += 1
                self.states[clone].len = self.states[p].len + 1
                self.states[clone].next = self.states[q].next.copy()
                self.states[clone].link = self.states[q].link

                while p != -1 and self.states[p].next.get(c) == q:
                    self.states[p].next[c] = clone
                    p = self.states[p].link
                
                self.states[q].link = self.states[cur].link = clone
        
        self.last = cur

@guidance(stateless=True, dedent=False)
def substring(lm, s):
    suffix_automaton = SuffixAutomaton(s)
    node_cache = {}
    state_stack = [0]  # Start with the initial state index (0) on the stack

    # Loop as long as there are states on the stack
    while state_stack:
        state_ind = state_stack[-1]  # Check the state on the top of the stack

        state = suffix_automaton.states[state_ind]

        # If we have already computed the result for this state, skip it
        if state_ind in node_cache:
            state_stack.pop()
            continue

        # If the state is a leaf node, meaning no outgoing edges (is an end of some suffix)
        if not state.next:
            node_cache[state_ind] = string("")  # Leaf nodes represent empty string suffixes
            state_stack.pop()
            continue

        # If there's an unprocessed child, add it to the stack
        unprocessed_children = [next_state for next_state in state.next.values() if next_state not in node_cache]
        if unprocessed_children:
            state_stack.extend(unprocessed_children)
        else:
            # Once all children are processed, create the node for this state
            options = [string(c) + node_cache[state.next[c]] for c in state.next]
            node_cache[state_ind] = optional(select(options, skip_checks=True)) if len(options) > 1 else optional(options[0])
            state_stack.pop()

    return lm + node_cache[0]

# @guidance(stateless=True, dedent=False)
# def substring(s):
#     a = SuffixAutomaton(s)
#     return _rec_substring(a, 0, {})

# def _rec_substring(suffix_automaton, state_ind, node_cache):
#     if state_ind in node_cache:
#         return node_cache[state_ind]

#     state = suffix_automaton.states[state_ind]

#     if len(state.next) == 0:
#         return string("")

#     options = []
#     for c in state.next:
#         options.append(string(c) + _rec_substring(suffix_automaton, state.next[c], node_cache))
    
#     if len(options) == 1:
#         node = optional(options[0])
#     else:
#         node = optional(select(options, skip_checks=True))
    
#     node_cache[state_ind] = node
#     return node