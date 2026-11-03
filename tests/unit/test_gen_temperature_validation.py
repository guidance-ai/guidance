import pytest

from guidance import gen, models


@pytest.mark.parametrize("temperature", ["high", True, [0.5]])
def test_non_number_temperature_is_named(temperature):
    lm = models.Mock()

    with pytest.raises(ValueError, match="`temperature` must be a number or None"):
        lm + gen(temperature=temperature, max_tokens=3)


def test_negative_temperature_is_named():
    lm = models.Mock()

    with pytest.raises(ValueError, match="`temperature` must be non-negative"):
        lm + gen(temperature=-0.5, max_tokens=3)


@pytest.mark.parametrize("temperature", [None, 0, 0.7])
def test_valid_temperatures_still_generate(temperature):
    lm = models.Mock()

    result = lm + gen(temperature=temperature, max_tokens=3)

    assert result is not None
