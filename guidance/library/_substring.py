import guidance
from ._prefix_tree import prefix_tree

@guidance(stateless=True, dedent=False)
def substring(lm, s):
    
    # build a prefix tree over 
    strings = [s[i:] for i in range(len(s))]
    return lm + prefix_tree(strings, partial_matches=True)