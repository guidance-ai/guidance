"""Elementary benchmarking for `guidance` development purposes.

`guidance` lives in a fast paced LLM environment, has complex dependencies and is tricky to implement.
These benchmarks are designed to focus on key use cases, where regressions can create havoc.

General guidelines:
- Simplicity first, then customization - reproducibility by the community is encouraged
- Everything takes forever - allow a pathway to scale horizontally
- Goalposts shift - some of the code for benchmarking will change frequently and that's okay
"""

# TODO(nopdive): Integrate powerlift for benchmarking backend
# TODO(nopdive): Implement langchain chat extraction task
# TODO(nopdive): Enable docker containers to execute benchmarking easily
