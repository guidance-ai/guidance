#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Guidance Contributors.
# Distributed under the terms of the Modified BSD License.

import pytest

from ..stitch import StitchWidget


def test_example_creation_blank():
    w = StitchWidget()
    assert w.kernelmsg == ""
    assert w.clientmsg == ""
    assert w.srcdoc == "<p>srcdoc should be defined by the user</p>"
