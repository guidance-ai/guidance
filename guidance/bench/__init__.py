"""Elementary benchmarking for `guidance` development purposes.

`guidance` lives in a fast paced LLM environment, has complex dependencies and is tricky to implement.
These benchmarks are designed to focus on key use cases, where regressions can create havoc.

General guidelines:
- Simplicity first, then customization - reproducibility by the community is encouraged
- Everything takes forever - allow a pathway to scale horizontally
- Goalposts shift - some of the code for benchmarking will change frequently and that's okay

Implementation:

The `bench` function is provided for no frills benchmarking that is designated for
automated testing.

For customization, we provide a notebook demonstration of how to run custom benchmarks
that are near mirror versions of what is available in the `bench` function provided.

Not implemented yet, but we intend to provide an avenue of running the benchmarks via
docker containers that have GPU resourcing to scale horizontally.
"""

from guidance.bench._powerlift import (
    retrieve_langchain,
    langchain_chat_extract_runner,
    langchain_chat_extract_filter_template,
)
from guidance.bench._api import bench

# TODO(nopdive): Enable docker containers to execute benchmarking easily
