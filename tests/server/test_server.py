import json
import multiprocessing
import time
from typing import List, Union

import pytest
import requests
from jsonschema import validate

from guidance import Server, gen, models
from guidance.library import json as gen_json
from guidance.library._json import _to_compact_json

# We spin out a separate process, and it
# has to start up and get ready to
# respond to requests. Just waiting is
# not ideal, but is the simplest option
PROCESS_DELAY_SECS = 90


def server_process(*, mock_string: Union[str, List[str]] = ""):
    byte_patterns = []
    if isinstance(mock_string, str):
        byte_patterns = [f"<s>{mock_string}<s>".encode()]
    else:
        for s in mock_string:
            byte_patterns.append(f"<s>{s}<s>".encode())
    lm = models.Mock(byte_patterns=byte_patterns)

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


def test_remote_mock_gen():
    with ServerContext(mock_string=""):
        m = models.Model("http://localhost:8392", api_key="SDFSDF")
        m2 = m + "A story." + gen("test", max_tokens=20)
        assert len(str(m2)) > 20, "The model didn't generate enough data"


def test_remote_mock_gen_bad_auth():
    with ServerContext(mock_string=""):
        m = models.Model("http://localhost:8392", api_key="FDSFDS")

        with pytest.raises(requests.exceptions.HTTPError) as http_err:
            _ = m + "A story." + gen("test", max_tokens=20)
        assert http_err.value.response.status_code == 401
        assert http_err.value.response.text == '{"detail":"Invalid API key"}'


def test_return_mock_string():
    my_string = "My roundtrip"
    with ServerContext(mock_string=my_string):
        m = models.Model("http://localhost:8392", api_key="SDFSDF")
        m2 = m + gen(max_tokens=20, name="captured")
        assert m2["captured"].startswith(my_string)


@pytest.mark.parametrize(
    "target_obj",
    [
        dict(my_list=None),
        dict(my_list=dict(my_str="a", next=None)),
        dict(my_list=dict(my_str="a", next=dict(my_str="b", next=None))),
    ],
)
def test_remote_gen_json(target_obj):
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

    print(f"target_obj={_to_compact_json(target_obj)}")

    with ServerContext(mock_string=[_to_compact_json(target_obj)]):
        m = models.Model("http://localhost:8392", api_key="SDFSDF")
        m += gen_json(schema=schema_obj, name="my_json_string")
        print(f"Raw: {m['my_json_string']}")

        my_obj = json.loads(m["my_json_string"])
        print(f"Received object: {json.dumps(my_obj, indent=4)}")
        validate(my_obj, schema_obj)
        assert my_obj == target_obj


def test_remote_list_append():
    with ServerContext(mock_string=""):
        lm = models.Model("http://localhost:8392", api_key="SDFSDF")
        for _ in range(3):
            lm += gen("my_list", list_append=True, stop="a") + "a"
        assert isinstance(lm["my_list"], list)
        assert len(lm["my_list"]) == 3
