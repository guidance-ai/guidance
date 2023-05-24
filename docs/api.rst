.. currentmodule:: guidance

API Reference
=============
This page contains the API reference for public objects and functions in Guidance.


.. _Program_api:

Program creation (a.k.a. `guidance(program_string)`)
-----------
.. autosummary::
    :toctree: generated/

    guidance.Program


.. _library_api:

library
----------
.. autosummary::
    :toctree: generated/

    guidance.library.add
    guidance.library.assistant
    guidance.library.await_
    guidance.library.block
    guidance.library.break_
    guidance.library.each
    guidance.library.equal
    guidance.library.gen
    guidance.library.geneach
    guidance.library.if_
    guidance.library.role
    guidance.library.select
    guidance.library.set
    guidance.library.shell
    guidance.library.strip
    guidance.library.subtract
    guidance.library.system
    guidance.library.unless
    guidance.library.user

.. _llms_api:

llms
-----
.. autosummary::
    :toctree: generated/

    guidance.llms.OpenAI
    guidance.llms.Transformers
    guidance.llms.transformers.LLaMA
    guidance.llms.transformers.MPT
    guidance.llms.transformers.StableLM
    guidance.llms.transformers.Vicuna
    guidance.llms.transformers.Koala