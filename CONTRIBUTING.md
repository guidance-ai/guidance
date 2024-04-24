# Contributing

This Project welcomes contributions, suggestions, and feedback. All contributions, suggestions, and feedback you submitted are accepted under the [Project's license](./LICENSE.md). You represent that if you do not own copyright in the code that you have the authority to submit it under the [Project's license](./LICENSE.md). All feedback, suggestions, or contributions are not confidential.

The Project abides by the Organization's [code of conduct](https://github.com/guidance-ai/governance/blob/main/CODE-OF-CONDUCT.md) and [trademark policy](https://github.com/guidance-ai/governance/blob/main/TRADEMARKS.md).

# Development Notes

We welcome contributions to `guidance`, and this document exists to provide useful information contributors.

## Developer Setup

The quickest way to get started is to run (in a fresh environment):
```bash
pip install -e .[all,test]
```
which should bring in all of the basic required dependencies.
Note that if you want to use GPU acceleration, then you will need to do whatever is required to allow `torch` and `llama-cpp` to access your GPU too.

There are sometimes difficulties configuring Rust during the pip installs.
If you encounter such issues, then one work around (if you use Anaconda) is to create your environment along the lines of
```bash
conda create -n guidance-312 python=3.12 rust
```
In our experience, this has been a little more reliable.
Similarly, to get GPU support, we have found that (after activating the environment) running
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```
works best.
However, if you have your own means of installing Rust and CUDA, you should be able to continue using those.

## Running Tests

Because we run tests on GPU-equipped machines and also tests which call LLM endpoints, approval is required before our GitHub workflows will run on external Pull Requests.
To run a basic test suite locally, we suggest:
```bash
python -m pytest -m "not (needs_credentials or use_gpu or server)" ./tests/
```
which runs our basic test suite.
Where an LLM is required, this will default to using GPT2 on the CPU.
To change that default, run
```bash
python -m pytest -m "not (needs_credentials or use_gpu or server)" --selected_model <MODELNAME> ./tests/
```
where `<MODELNAME>` is taken from the `AVAILABLE_MODELS` dictionary defined in `conftest.py`.

## Adding LLMs to the test matrix

Our

## Linting

We run `black` on our codebase, and plan to turn on enforcement of this in the GitHub workflows soon.


---
Part of MVG-0.1-beta.
Made with love by GitHub. Licensed under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/).