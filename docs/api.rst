.. currentmodule:: guidance

API Reference
=============
This page contains the API reference for public objects and functions in Guidance.


.. _functions_api:

functions
---------
.. autosummary::
    :toctree: generated/

    guidance.gen
    guidance.select


.. _contexts_api:

context blocks
--------------
.. autosummary::
    :toctree: generated/

    guidance.instruction
    guidance.system
    guidance.user
    guidance.assistant


.. _models_api:

models
------
.. autosummary::
    :toctree: generated/

    guidance.models.Model
        guidance.models.Instruct
        guidance.models.Chat
    guidance.models.LlamaCpp
    guidance.models.Transformers
    guidance.models.Remote
        guidance.models.VertexAI
        guidance.models.GoogleAI
        guidance.models.OpenAI
        guidance.models.LiteLLM
        guidance.models.Cohere
        guidance.models.Anthropic
