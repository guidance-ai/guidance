import importlib
import typing


def optional_import(name: str):
    msg = f"Import of package {name} failed."
    module = None
    try:
        module = importlib.import_module(name)
    except ImportError:
        if typing.TYPE_CHECKING:
            raise ImportError(msg)
    return module
