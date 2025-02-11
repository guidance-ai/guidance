#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Guidance Contributors.
# Distributed under the terms of the Modified BSD License.

"""
Stitch Widget that allows bidirectional comms from Jupyter and JavaScript.
"""

from ipywidgets import DOMWidget
from traitlets import Unicode
from ._frontend import module_name, module_version


class StitchWidget(DOMWidget):
    """Widget that purely handles communication between an iframe and kernel via postMessage."""

    _model_name = Unicode('StitchModel').tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode('StitchView').tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)

    kernelmsg = Unicode("").tag(sync=True)
    clientmsg = Unicode("").tag(sync=True)
    srcdoc = Unicode("<p>srcdoc should be defined by the user</p>").tag(sync=True)
    initial_height = Unicode("1px").tag(sync=True)
    initial_width = Unicode("1px").tag(sync=True)
    initial_border = Unicode("0").tag(sync=True)

    # NOTE(nopdive): Should we sync or not? There are overheads when we deal with bandwidth on real time applications.
    state = Unicode("").tag(sync=True)