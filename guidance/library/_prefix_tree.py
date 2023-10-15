import guidance
from ._select import select
from ._string import string

@guidance(stateless=True, dedent=False)
def prefix_tree(lm, strings, partial_matches=False):

    # group the strings by their starting character
    char_groups = {}
    for s in strings:
        if len(s) > 0:
            if s[0] not in char_groups:
                char_groups[s[0]] = []
            char_groups[s[0]].append(s[1:])
    
    # recursively build the tree
    suboptions = [string(k) + prefix_tree(v, partial_matches=partial_matches) for k,v in char_groups.items()] 
    
    if len(suboptions) == 0:
        return string("")
    
    elif partial_matches:
        suboptions.append("")

    return lm + select(suboptions)