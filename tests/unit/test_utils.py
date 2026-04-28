from pathlib import Path

import pytest

from guidance._utils import bytes_from


def test_bytes_from_file_uri_respects_allow_local(tmp_path: Path) -> None:
    local_file = tmp_path / "secret.bin"
    local_file.write_bytes(b"local bytes")

    assert bytes_from(local_file.as_uri(), allow_local=True) == b"local bytes"

    with pytest.raises(Exception, match="Unable to load bytes"):
        bytes_from(local_file.as_uri(), allow_local=False)