# Development Notes

We welcome contributions to `guidance`, and this document exists to provide useful information contributors.

## Developer Setup

The quickest way to get started is to run (in a fresh environment):
```bash
pip install -e .[all,test]
```
which should bring in all of the basic required dependencies.
Note that if you want to use GPU acceleration, then you will need to do whatever is required to allow `torch` and `llama-cpp` to access your GPU too.

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

## Linting

We run `black` on our codebase, and plan to turn on enforcement of this in the GitHub workflows soon.