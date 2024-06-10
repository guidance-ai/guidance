# Testing

## Organisation

The tests are arranged into the following directories:

- `unit` tests do not depend on LLMs (but may use `model.Mock`)
- `model_integration` tests should run on any (local) model, supplied by the `selected_model` fixture
- `model_specific` tests are for isolating particular issues with individual LLMs
- `need_credentials` tests are for tests which need access to various credentials (mainly `Grammarless` models)
- `notebook` tests are for notebooks

The `model_specific` tests should make use of the `selected_model` machinery, but skip themselves if the appropriate model is not supplied.
A sample means of achieving this:

```python
@pytest.fixture(scope="module")
def phi3_model(selected_model, selected_model_name):
    if selected_model_name in ["transformers_phi3cpu_mini_4k_instruct"]:
        return selected_model
    else:
        pytest.skip("Requires Phi3 model")
```