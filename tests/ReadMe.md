# Testing

## Organisation

The tests are arranged into the following directories:

- `unit` tests do not depend on LLMs (but may use `model.Mock`)
- `modelintegration` tests should run on any (local) model, supplied by the `selected_model` fixture
- `modelspecific` tests are for isolating particular issues with individual LLMs
- `grammarless` tests are for `Grammarless` models - those which depend on a remote endpoint which is not Guidance aware
- `notebook` tests are for notebooks

The `modelspecific` tests should make use of the `selected_model` machinery, but skip themselves if the appropriate model is note supplied.
A sample means of achieving this:

```python
@pytest.fixture(scope="module")
def phi3_model(selected_model, selected_model_name):
    if selected_model_name in ["transformers_phi3cpu_mini_4k_instruct"]:
        return selected_model
    else:
        pytest.skip("Requires Phi3 model")
```