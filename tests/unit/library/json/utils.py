import json
from functools import partial
from json import dumps as json_dumps
from json import loads as json_loads
from typing import Any

from jsonschema import validate

from guidance import json as gen_json
from guidance.library._json import JSONSchema

from ....utils import check_match_failure as _check_match_failure
from ....utils import check_run_with_temperature
from ....utils import generate_and_check as _generate_and_check


def generate_and_check(
    target_obj: Any,
    schema_obj: str | JSONSchema,
    desired_temperature: float | None = None,
):
    if isinstance(schema_obj, str):
        schema_obj = json_loads(schema_obj)

    # Sanity check what we're being asked
    validate(instance=target_obj, schema=schema_obj)
    prepared_json = json_dumps(target_obj)
    assert json.loads(prepared_json) == target_obj

    # Now test that the grammar can recognize and generate prepared_json
    # We partial in the grammar_callable
    if desired_temperature is not None:
        grammar_callable = partial(gen_json, schema=schema_obj, temperature=desired_temperature)
    else:
        grammar_callable = partial(gen_json, schema=schema_obj)

    lm = _generate_and_check(
        grammar_callable,
        test_string=prepared_json,
    )
    check_run_with_temperature(lm, desired_temperature)


def check_match_failure(
    *,
    bad_string: str,
    good_bytes: bytes | None = None,
    failure_byte: bytes | None = None,
    allowed_bytes: set[bytes] | None = None,
    schema_obj: str | JSONSchema,
):
    grammar = gen_json(schema=schema_obj)

    _check_match_failure(
        bad_string=bad_string,
        good_bytes=good_bytes,
        failure_byte=failure_byte,
        allowed_bytes=allowed_bytes,
        grammar=grammar,
    )
