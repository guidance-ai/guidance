import json
import multiprocessing
import time

import pytest
import requests
from jsonschema import validate

from guidance import Server, gen, models
from guidance.library import gen_json

from .utils import get_model


def server_process():
    gpt2 = get_model(f"transformers:gpt2")

    server = Server(gpt2, api_key="SDFSDF")
    server.run(port=8392)


@pytest.fixture
def running_server():
    p = multiprocessing.Process(target=server_process)
    p.start()
    time.sleep(60)
    yield p
    p.terminate()
    time.sleep(60)
    assert not p.is_alive(), "server_process failed to terminate"


def test_remote_mock_gen(running_server):
    m = models.Model("http://localhost:8392", api_key="SDFSDF")
    m2 = m + "A story." + gen("test", max_tokens=20)
    assert len(str(m2)) > 20, "The model didn't generate enough data"


def test_remote_mock_gen_bad_auth(running_server):
    m = models.Model("http://localhost:8392", api_key="FDSFDS")

    with pytest.raises(requests.exceptions.HTTPError) as http_err:
        _ = m + "A story." + gen("test", max_tokens=20)
    assert http_err.value.response.status_code == 401
    assert http_err.value.response.text == '{"detail":"Invalid API key"}'


def test_remote_gen_json(running_server):
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
    m = models.Model("http://localhost:8392", api_key="SDFSDF")

    objs = [
        dict(my_list=dict(my_str="a", next=None)),
        dict(my_list=dict(my_str="a", next=dict(my_str="b", next=None))),
        dict(
            my_list=dict(
                my_str="a", next=dict(my_str="b", next=dict(my_str="c", next=None))
            )
        ),
    ]
    # Sanity check samples
    for obj in objs:
        validate(obj, schema_obj)

    m += "Produce the next object in this sequence:\n"
    for obj in objs:
        m += json.dumps(obj, separators=(",", ":")) + "\n"
    m += gen_json(schema_obj, name="my_json_string")

    my_obj = json.loads(m["my_json_string"])
    print(json.dumps(my_obj, indent=4))

    validate(my_obj, schema_obj)
