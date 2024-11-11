""" Rendering environment detection.

Detection logic is inspired from both plotly and interpretml environment detection.
- https://github.com/plotly/plotly.py
- https://github.com/interpretml/interpret
"""
# TODO(nopdive): Major cloud providers implemented and manually verified.

import os
from pydantic import BaseModel


class EnvFlags(BaseModel):
    """Environment flags - such as if we're in a notebook or cloud."""

    is_notebook: bool = False
    is_cloud: bool = False


class Environment:
    """ Capabilities based environment detection."""

    def __init__(self):
        """ Initializes.

        This will immediately check for which environments are detected.
        """
        self._flags = EnvFlags()
        self._checks = {
            "vscode": _detect_vscode,
            "ipython-zmq": _detect_ipython_zmq,
            "ipython": _detect_ipython,
        }
        envs = []
        for name, update_flags in self._checks.items():
            if update_flags(self._flags):  # in-place operation on flags
                envs.append(name)
        self._detected_envs = envs


    @property
    def detected_envs(self) -> list[str]:
        """ Detected environments (i.e. vscode, ipython-zmq).

        Returns:
            Detected environment names.
        """
        return self._detected_envs


    def is_notebook(self) -> bool:
        """ Determines if the python process is in a notebook.

        Returns:
            True if in notebook.
        """
        return self._flags.is_notebook

    def is_cloud(self) -> bool:
        """ Determines if the python process is in a cloud provider.

        Returns:
            True if in notebook.
        """
        return self._flags.is_cloud

    def is_terminal(self) -> bool:
        """ Determines if the python process not in a notebook (we assume terminal).

        Returns:
            True if in terminal.
        """
        return not self._flags.is_notebook


def _detect_vscode(flags: EnvFlags) -> bool:
    """Detects if called in a vscode process.

    Args:
        flags: Inplace flags to be set.

    Returns:
        True if in vscode environment.
    """
    # NOTE: We don't flag is_notebook since this will be picked up by ipython-zmq here.
    found = "VSCODE_PID" in os.environ
    return found


def _detect_ipython(flags: EnvFlags) -> bool:
    """Detects if called in an IPython environment.
    Mostly derived from stackoverflow below:
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    Args:
        flags: Inplace flags to be set.

    Returns:
        True if in IPython environment.
    """
    found = False
    try:
        from IPython import get_ipython
        found = get_ipython() is not None
    except (NameError, ImportError):  # pragma: no cover
        pass
    return found


def _detect_ipython_zmq(flags: EnvFlags) -> bool:
    """Detects if in an IPython environment using ZMQ (i.e. notebook/qtconsole/lab).

    Mostly derived from stackoverflow below:
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook/24937408

    Args:
        flags: Inplace flags to be set.

    Returns:
        True if called in IPython notebook or qtconsole.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            found = True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            found = False  # Terminal running IPython
        else:
            found = False  # Other type (?)
    except (NameError, ImportError):  # pragma: no cover
        found = False  # Probably standard Python interpreter

    flags.is_notebook |= found
    return found