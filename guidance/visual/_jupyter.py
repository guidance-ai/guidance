""" Jupyter specific utilities."""


import weakref
from typing import Callable, Any, Tuple
import logging
try:
    from IPython import get_ipython
except ImportError:
    pass

logger = logging.getLogger(__name__)

IPythonCallback = Callable[[Any], None]

# TODO(nopdive): Connect cell handling again to renderer.


def ipy_handle_event_once(cb: IPythonCallback, event_name: str) -> IPythonCallback:
    ipy = get_ipython()

    if ipy is None:
        return None
    
    def cb_closure(msg):
        cb(info=msg)
        ipy.events.unregister(event_name, cb_closure)
    ipy.events.register(event_name, cb_closure)
    
    return cb_closure