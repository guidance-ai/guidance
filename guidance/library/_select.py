# select is a special core function that is defined using low-level grammar primitives
from .._grammar import _select as select

# TODO: allow for returning the probabilites of the selected item
# TODO: also the full probabilites distribution over all items. We can implement this using the prob of the selected item by repeating the call, removing the selected item each time