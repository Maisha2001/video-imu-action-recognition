from io import BytesIO
from zipfile import ZipFile

import pytest

import fetch_data
from activity_data import file_hash


class Response(BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload))}


def test_checked_download_is_reused_without_network(tmp_path, monkeypatch):
    source = tmp_path / "source.zip"
    with ZipFile(source, "w") as bundle:
        bundle.writestr("example.txt", "fixture")
    payload = source.read_bytes()
    monkeypatch.setitem(fetch_data.HASHES, "Inertial.zip", file_hash(source))
    calls = []

    def download(url, timeout):
        calls.append(url)
        return Response(payload)

    monkeypatch.setattr(fetch_data, "urlopen", download)
    output = tmp_path / "download"
    first = fetch_data.fetch("Inertial.zip", output)
    assert fetch_data.fetch("Inertial.zip", output) == first
    assert first.read_bytes() == payload
    assert len(calls) == 1


def test_bad_download_is_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, "urlopen", lambda *args, **kwargs: Response(b"invalid"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_data.fetch("Inertial.zip", tmp_path)
    assert not (tmp_path / "Inertial.zip").exists()
    assert not (tmp_path / "Inertial.zip.partial").exists()


def test_existing_bad_archive_is_preserved(tmp_path):
    path = tmp_path / "Inertial.zip"
    path.write_bytes(b"user file")
    with pytest.raises(ValueError, match="Existing archive checksum"):
        fetch_data.fetch("Inertial.zip", tmp_path)
    assert path.read_bytes() == b"user file"


def test_interrupted_download_resumes_at_saved_offset(tmp_path, monkeypatch):
    partial = tmp_path / "RGB.zip.partial"
    calls = []

    def interrupted(request, timeout):
        calls.append(request.get_header("Range"))
        if len(calls) == 1:
            return Response(b"first", headers={"Content-Length": "11"})
        return Response(b"second", 206, {"Content-Range": "bytes 5-10/11", "Content-Length": "6"})

    monkeypatch.setattr(fetch_data, "urlopen", interrupted)
    fetch_data.download("https://example.test/data", partial)
    assert partial.read_bytes() == b"firstsecond"
    assert calls == [None, "bytes=5-"]


def test_ignored_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    partial = tmp_path / "RGB.zip.partial"
    partial.write_bytes(b"prefix")
    monkeypatch.setattr(fetch_data, "urlopen", lambda *args, **kwargs: Response(b"complete"))
    fetch_data.download("https://example.test/data", partial)
    assert partial.read_bytes() == b"complete"


def test_wrong_range_does_not_corrupt_partial_file(tmp_path, monkeypatch):
    partial = tmp_path / "RGB.zip.partial"
    partial.write_bytes(b"prefix")
    monkeypatch.setattr(fetch_data, "urlopen", lambda *args, **kwargs: Response(
        b"bad", 206, {"Content-Range": "bytes 3-5/6"}))
    with pytest.raises(ValueError, match="unexpected byte range"):
        fetch_data.download("https://example.test/data", partial)
    assert partial.read_bytes() == b"prefix"
