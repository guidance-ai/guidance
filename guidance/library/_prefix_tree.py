import guidance
from .._grammar import select, string

@guidance(stateless=True, dedent=False)
def prefix_tree(lm, strings, partial_matches=False):

    if len(strings) == 0:
        return lm

    # group the strings by their starting character
    char_groups = {}
    for s in strings:
        if len(s) > 0:
            if s[0] not in char_groups:
                char_groups[s[0]] = []
            char_groups[s[0]].append(s[1:])
    
    # enable any empty followup if partial matches are allowed
    if partial_matches:
        char_groups[""] = []
    
    # recursively build the tree
    suboptions = [string(k) + prefix_tree(v, partial_matches=partial_matches) for k,v in char_groups.items()]

    return lm + select(suboptions, skip_checks=True) # we skip normal type checks for speed