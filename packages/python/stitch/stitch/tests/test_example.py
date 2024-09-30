#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Guidance Contributors.
# Distributed under the terms of the Modified BSD License.

import pytest

from ..stitch import StitchWidget


def test_example_creation_blank():
    w = StitchWidget()
    assert w.value == 'Hello World'
