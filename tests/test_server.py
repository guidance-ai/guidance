import json
import multiprocessing
import time

from typing import List, Union

import pytest
import requests
from jsonschema import validate

from guidance import Server, gen, models
from guidance.library import gen_json

from .utils import to_compact_json

PROCESS_DELAY_SECS = 10


def server_process(*, mock_string: Union[str, List[str]] = ""):
    if isinstance(mock_string, str):
        prepared_string = f"<s>{mock_string}".encode()
    else:
        prepared_string = []
        for s in mock_string:
            prepared_string.append(f"<s>{s}".encode())
    lm = models.Mock(byte_patterns=prepared_string)

    temp_lm = lm + gen()
    print(f"=====Plain gen output: {temp_lm}=====")

    server = Server(lm, api_key="SDFSDF")
    server.run(port=8392)


class ServerContext:
    def __init__(self, mock_string: str):
        self._process = multiprocessing.Process(
            target=server_process, kwargs=dict(mock_string=mock_string)
        )

    def __enter__(self):
        self._process.start()
        time.sleep(PROCESS_DELAY_SECS)
        return None

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._process.terminate()
        time.sleep(PROCESS_DELAY_SECS)
        return False  # We don't handle exceptions


@pytest.fixture
def running_server(*, mock_string: str = ""):
    p = multiprocessing.Process(
        target=server_process, kwargs=dict(mock_string=mock_string)
    )
    p.start()
    time.sleep(PROCESS_DELAY_SECS)
    yield p
    p.terminate()
    time.sleep(PROCESS_DELAY_SECS)
    assert not p.is_alive(), "server_process failed to terminate"


def n_test_remote_mock_gen(running_server):
    m = models.Model("http://localhost:8392", api_key="SDFSDF")
    m2 = m + "A story." + gen("test", max_tokens=20)
    assert len(str(m2)) > 20, "The model didn't generate enough data"


def n_test_remote_mock_gen_bad_auth(running_server):
    m = models.Model("http://localhost:8392", api_key="FDSFDS")

    with pytest.raises(requests.exceptions.HTTPError) as http_err:
        _ = m + "A story." + gen("test", max_tokens=20)
    assert http_err.value.response.status_code == 401
    assert http_err.value.response.text == '{"detail":"Invalid API key"}'


def n_test_return_mock_string():
    my_string = "My roundtrip"
    with ServerContext(mock_string=my_string):
        m = models.Model("http://localhost:8392", api_key="SDFSDF")
        m2 = m + gen(max_tokens=10, name="captured")
        assert m2["captured"].startswith(my_string)


@pytest.mark.parametrize(
    ["target_obj", "mock_strings"],
    [
        (dict(my_list=None), ['{"my_list":n']),
        (
            dict(my_list=dict(my_str="a", next=None)),
            [
                '{"my_list":{',
                '{"my_list":{"my_str":"a"',
                '{"my_list":{"my_str":"a","next":n',
            ],
        ),
        (
            dict(my_list=dict(my_str="a", next=dict(my_str="b", next=None))),
            [
                '{"my_list":{',
                '{"my_list":{"my_str":"a"',
                '{"my_list":{"my_str":"a","next":{',
                '{"my_list":{"my_str":"a","next":{"my_str":"b"',
                '{"my_list":{"my_str":"a","next":{"my_str":"b","next":n',
            ],
        ),
    ],
)
def test_remote_gen_json(target_obj, mock_strings: List[str]):
    # To explain the arguments:
    # target_obj is what we're trying to generate
    # mock_strings is the sequence of partial strings model.Mock
    # needs to produce. Basically, these are the sequence of
    # strings which reach the next 'unambiguous point' in the
    # grammar. Also recall that if it can't use a supplied
    # string, model.Mock() will just generate random bytes
    #
    # As an example, consider the two inputs:
    # target_obj = dict(my_list=dict(my_str="a", next=None))
    # mock_strings =
    #        [
    #            '{"my_list":{',
    #            '{"my_list":{"my_str":"a"',
    #            '{"my_list":{"my_str":"a","next":n',
    #        ]
    # When the remote model starts generating, the string:
    # {"my_list":
    # is going to be forced by the JSON schema we've supplied.
    # So the first entry in mock_strings has to be this, plus
    # the opening brace which indicates we do have a list entry
    # Once the remote model knows this, then it can automatically
    # generate up to:
    # {"my_list":{"my_str":"
    # so mock_strings[1] needs to append 'a"' in order to get
    # to the next unambiguous point.
    # Finally, everything up to
    # {"my_list":{"my_str":"a","next":
    # can be appended unambiguously, so mock_string[2] just
    # needs to append 'n' to force the generation of 'null' to
    # terminate the list
    schema = """
{
    "$defs": {
        "A": {
            "properties": {
                "my_str": {
                    "default": "me",
                    "title": "My Str",
                    "type": "string"
                },
                "next": {
                    "anyOf": [
                        {
                            "$ref": "#/$defs/A"
                        },
                        {
                            "type": "null"
                        }
                    ]
                }
            },
            "type": "object"
        }
    },
    "type": "object",
    "properties": {
        "my_list": {
            "anyOf": [
                {
                    "$ref": "#/$defs/A"
                },
                {
                    "type": "null"
                }
            ]
        }
    }
}
        """
    schema_obj = json.loads(schema)

    # Sanity check input
    validate(target_obj, schema_obj)

    print(f"target_obj={to_compact_json(target_obj)}")

    with ServerContext(mock_string=mock_strings):
        m = models.Model("http://localhost:8392", api_key="SDFSDF")
        m += ""
        m += gen_json(schema_obj, name="my_json_string")
        print(f"Raw: {m['my_json_string']}")

        my_obj = json.loads(m["my_json_string"])
        print(f"Received object: {json.dumps(my_obj, indent=4)}")
        validate(my_obj, schema_obj)
        assert my_obj == target_obj
