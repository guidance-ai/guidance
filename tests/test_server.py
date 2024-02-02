import multiprocessing
import time

import pytest
import requests

from guidance import Server, gen, models


def server_process():
    mistral = models.Mock()

    server = Server(mistral, api_key="SDFSDF")
    server.run(port=8392)


@pytest.fixture
def running_server():
    p = multiprocessing.Process(target=server_process)
    p.start()
    time.sleep(10)
    yield p
    p.terminate()
    time.sleep(10)
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
