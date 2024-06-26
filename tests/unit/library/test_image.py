import pytest

from urllib.error import HTTPError
from guidance import models, image


def test_local_image(local_image_path):
    model = models.Mock()
    model += image(local_image_path)

    assert str(model).startswith("<|_image:")


def test_local_image_not_found():
    model = models.Mock()
    with pytest.raises(FileNotFoundError):
        model += image("not_found.jpg")


def test_remote_image(remote_image_url):
    model = models.Mock()
    model += image(remote_image_url)

    assert str(model).startswith("<|_image:")


def test_remote_image_not_found():
    model = models.Mock()
    with pytest.raises(HTTPError):
        model += image("https://example.com/not_found.jpg")


def test_image_from_bytes(local_image_bytes):
    model = models.Mock()
    model += image(local_image_bytes)
    assert str(model).startswith("<|_image:")
