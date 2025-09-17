"""Debug utilities for the guidance widget."""

import logging
from typing import Optional

from .registry import get_renderer
from .visual._renderer import AutoRenderer, JupyterWidgetRenderer

logger = logging.getLogger(__name__)


def enable_widget_debug() -> None:
    """Enable debug mode in the guidance widget.

    This will start capturing all messages sent to the widget for later inspection.
    Call dump_widget_debug() to dump the captured data to a string.

    Example:
        >>> import guidance
        >>> guidance.enable_widget_debug()
        >>> # ... run your guidance code ...
        >>> guidance.dump_widget_debug()
    """
    renderer = get_renderer()
    if isinstance(renderer, JupyterWidgetRenderer):
        renderer.enable_debug()
    elif isinstance(renderer, AutoRenderer):
        # Check if the auto renderer has a JupyterWidgetRenderer inside
        if hasattr(renderer, "_renderer") and isinstance(renderer._renderer, JupyterWidgetRenderer):
            renderer._renderer.enable_debug()
        else:
            logger.warning(f"Auto renderer contains {type(renderer._renderer)} instead of JupyterWidgetRenderer")
    else:
        logger.warning(f"Debug mode only available with Jupyter widget renderer, got {type(renderer)}")


def dump_widget_debug() -> str | None:
    """Get captured debug data as a JSON string.

    Returns the captured widget messages and state as a JSON string,
    or None if no debug data is available.

    Example:
        >>> import guidance
        >>> guidance.enable_widget_debug()
        >>> # ... run your guidance code ...
        >>> debug_data = guidance.dump_widget_debug()
        >>> print(debug_data)  # inspect in notebook
    """
    renderer = get_renderer()
    if isinstance(renderer, JupyterWidgetRenderer):
        return renderer.get_debug_data()
    elif isinstance(renderer, AutoRenderer):
        # Check if the auto renderer has a JupyterWidgetRenderer inside
        if hasattr(renderer, "_renderer") and isinstance(renderer._renderer, JupyterWidgetRenderer):
            return renderer._renderer.get_debug_data()
        else:
            logger.warning(f"Auto renderer contains {type(renderer._renderer)} instead of JupyterWidgetRenderer")
            return None
    else:
        logger.warning(f"Debug dump only available with Jupyter widget renderer, got {type(renderer)}")
        return None


def clear_widget_debug() -> None:
    """Clear captured widget debug messages.

    Useful for starting fresh between different test runs.

    Example:
        >>> import guidance
        >>> guidance.enable_widget_debug()
        >>> # ... run some guidance code ...
        >>> guidance.clear_widget_debug()  # clear before next test
        >>> # ... run different guidance code ...
        >>> debug_data = guidance.dump_widget_debug()
    """
    renderer = get_renderer()
    if isinstance(renderer, JupyterWidgetRenderer):
        renderer.clear_debug_data()
    elif isinstance(renderer, AutoRenderer):
        # Check if the auto renderer has a JupyterWidgetRenderer inside
        if hasattr(renderer, "_renderer") and isinstance(renderer._renderer, JupyterWidgetRenderer):
            renderer._renderer.clear_debug_data()
        else:
            logger.warning(f"Auto renderer contains {type(renderer._renderer)} instead of JupyterWidgetRenderer")
    else:
        logger.warning(f"Clear debug only available with Jupyter widget renderer, got {type(renderer)}")


def widget_debug_info() -> None:
    """Print debug information about the current widget renderer setup.

    This function displays Python-side version info and requests version info
    from the JavaScript widget if available.
    """
    import platform
    import sys

    from .visual._environment import Environment

    renderer = get_renderer()
    env = Environment()

    print(f"Renderer type: {type(renderer)}")
    print(f"Environment is_notebook(): {env.is_notebook()}")
    print(f"Environment is_terminal(): {env.is_terminal()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.release()}")

    # Check stitch installation (Python side)
    try:
        import stitch

        version = getattr(stitch, "__version__", "unknown")
        print(f"Stitch (Python): {version}")
    except ImportError:
        print("Stitch (Python): Not installed")

    # Check guidance version
    try:
        import guidance

        guidance_version = getattr(guidance, "__version__", "unknown")
        print(f"Guidance version: {guidance_version}")
    except ImportError:
        guidance_version = "unknown"
        print("Guidance version: unknown")

    # Try to get widget version info if we have a JupyterWidgetRenderer
    widget_renderer = None
    if isinstance(renderer, JupyterWidgetRenderer):
        widget_renderer = renderer
        print("Widget debug mode should work!")
    elif isinstance(renderer, AutoRenderer):
        print(f"AutoRenderer inner type: {type(renderer._renderer)}")
        if hasattr(renderer, "_renderer") and isinstance(renderer._renderer, JupyterWidgetRenderer):
            widget_renderer = renderer._renderer
            print("Widget debug mode should work!")
        else:
            print("Widget debug mode not available - inner renderer is not JupyterWidgetRenderer")
    else:
        print("Widget debug mode not available - not using widget renderer")
