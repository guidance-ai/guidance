""" Jupyter specific utilities."""


from typing import Callable, Any, Tuple, Optional
import logging
from uuid import uuid4

try:
    from IPython import get_ipython
except ImportError:
    pass

logger = logging.getLogger(__name__)

IPythonCallback = Callable[[Any], None]


def ipy_handle_event_once(cb: IPythonCallback, event_name: str) -> Tuple[Optional[IPythonCallback], str]:
    ipy = get_ipython()
    cell_session_id = str(uuid4())

    if ipy is None:
        return None, ""
    
    def cb_closure(msg):
        cb(info=msg)
        ipy.events.unregister(event_name, cb_closure)
    ipy.events.register(event_name, cb_closure)
    
    return cb_closure, cell_session_id