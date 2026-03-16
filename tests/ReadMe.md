# Testing

## Organisation

The tests are arranged into the following directories:

- `unit` tests do not depend on LLMs (but may use `model.Mock`)
- `model_integration` tests should run on any (fully supported) model, supplied by the `selected_model` fixture
- `model_specific` tests are for isolating particular issues with individual LLMs
- `need_credentials` tests are for tests which need access to various credentials (mainly `Grammarless` models for endpoints without full Guidance support)
- `notebook` tests are for notebooks

The `model_specific` tests should make use of the `selected_model` machinery, but skip themselves if the appropriate model is not supplied.
A sample means of achieving this:

```python
@pytest.fixture(scope="module")
def phi3_model(selected_model, selected_model_name):
    if selected_model_name in ["transformers_phi3_mini_4k_instruct_cpu"]:
        return selected_model
    else:
        pytest.skip("Requires Phi3 model")
```

## Selecting a model

To select a particular model when running the tests, use the `--selected_model` command line option.
For example:

```bash
python -m pytest --selected_model transformers_gpt2_cpu ./tests/model_integration/
```

The allowed values for `--selected_model` are in the [`confest.py`](./conftest.py) file, and are defined in the `selected_model` function.
Alternatively, the `GUIDANCE_SELECTED_MODEL` environment variable can be used to override the default value for `--selected_model` (which can be useful when using a debugger).

### A Note on Credentials

As noted above the `need_credentials` tests are mainly for `Grammarless` models - those for remote endpoints which do not support Guidance grammars (there are a few exceptions, which is why the directory isn't simply named `grammarless`).
As endpoints with Guidance grammar support come online, their tests should *not* go in there; these should go into `model_integration` and `model_specific`, but will only be run in CI builds.
Similarly, some models (e.g. LLama3) require credentials in order to download their weights from Hugging Face.
These should be run through the `model_integration` and `model_specific` tests, but this run will happen from the CI build, and hence have credential access.

## Testing Goal

Ideally, when creating a new feature, most of the tests should go into the `unit` directory, and make use of `model.Mock` if needed.
These should always be able to be run with quite a minimal Guidance installation (have to add `pytest`, obviously).
These tests should be fast, and facilitate a developer experience build around running

```bash
pytest tests/unit
```
very frequently.

There should also be a handful of tests in `model_integration`, which should work with _any_ fully supported Guidance model.
Finally, if any model quirks are noted (and _especially_ if workarounds are required in the code), tests to characterise these should go into `model_specific`.

In this paradigm, no tests in `unit` or `model_integration` should be using `pytest.skip` (or its variants).
Those in `model_specific` will use `pytest.skip` for when the `selected_model` fixture is not of the appropriate type.
