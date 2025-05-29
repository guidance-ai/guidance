# Contributing

This Project welcomes contributions, suggestions, and feedback. All contributions, suggestions, and feedback you submitted are accepted under the [Project's license](./LICENSE.md). You represent that if you do not own copyright in the code that you have the authority to submit it under the [Project's license](./LICENSE.md). All feedback, suggestions, or contributions are not confidential.

The Project abides by the Organization's [code of conduct](https://github.com/guidance-ai/governance/blob/main/CODE-OF-CONDUCT.md) and [trademark policy](https://github.com/guidance-ai/governance/blob/main/TRADEMARKS.md).

# Development Notes

We welcome contributions to `guidance`, and this document exists to provide useful information contributors.

## Developer Setup

Start by creating a fresh environment with something similar to:
```bash
conda create --name guidancedev python=3.12
conda activate guidancedev
```

Install guidance (without CUDA):
```bash
python -m pip install -e .[all,test,bench,llamacpp,transformers]
```

Alternatively, install guidance with CUDA support. There are various ways to do this. We recommend:
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install -e .[all,test,bench,llamacpp,transformers]
```

## Running Tests

To run a basic test suite locally:
```bash
python -m pytest ./tests/
```
which runs our basic test suite.
Where an LLM is required, this will default to using GPT2 on the CPU.

To change that default, run
```bash
python -m pytest --selected_model <MODELNAME> ./tests/
```
where `<MODELNAME>` is taken from one of the selected_model_name options defined in `./tests/conftest.py`.

Alternatively, the default value for `--selected_model` can be set via the `GUIDANCE_SELECTED_MODEL` environment variable.
This may be useful when trying to use a debugger when running `pytest`, and setting the extra command line argument in the debugger configuration is tricky.
Just remember that the environment variable needs to be set _before_ starting PyCharm/VSCode etc.

## Adding LLMs to the test matrix

Our tests run on a variety of LLMs.
These fall into three categories: CPU-based, GPU-based and endpoint-based (which need credentials).

### New CPU or GPU-based models

Due to the limited resources of the regular GitHub runner machines, the LLM under test is a dimension of our test matrix (otherwise the GitHub runners will tend to run out of RAM and/or hard drive space).
New models should be configured in `conftest.py`.
The model will then be available via the `selected_model` fixture for all tests.
If you have a test which should only run for particular models, you can use the `selected_model_name` fixture to check, and call `pytest.skip()` if necessary.
An example of this is given in `test_llama_cpp.py`.

### New endpoint based models

If your model requires credentials, then those will need to be added to our GitHub repository as secrets.
The endpoint itself (and any other required information) should be configured as environment variables too.
When the test runs, the environment variables will be set, and can then be used to configure the model as required.
See `test_azureai_openai.py` for examples of this being done.

## Formatting & Linting

We use `ruff` to format our codebase.
To install the correct version, run `pip install -e .[dev]`.
You can then run `ruff format /path/to/modified/file.py` to format the code.
The path can be an entire directory, or omitted entirely to format all files beneath the current directory.
There are (rare) cases where manual formatting is preferable; for these [`ruff` provides pragmas for suppression](https://docs.astral.sh/ruff/formatter/#format-suppression).
To sort imports, use `ruff check --select I /path/to/modified/file.py`.
These commands are run (but not enforced *yet*) in the build.



---
Part of MVG-0.1-beta.
Made with love by GitHub. Licensed under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).
