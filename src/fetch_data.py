import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from activity_data import file_hash


SOURCES = {"Inertial.zip": "https://www.utdallas.edu/~kehtar/UTD-MAD/Inertial.zip",
           "RGB.zip": "https://www.utdallas.edu/~kehtar/UTD-MAD/RGB.zip"}
HASHES = {"Inertial.zip": "1900de05775c04626dee30b34638b79dd739d481786f0a788f2fbb00663abea4",
          "RGB.zip": "058dded078b2c21eff901d98fe0a86d5343d11e9d6c07081aaa3e3280a43052b"}


def download(url, partial):
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "video-imu-action-recognition/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=60) as response:
                append = response.status == 206 and offset > 0
                if response.status == 206 and not response.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected byte range")
                received = 0
                with partial.open("ab" if append else "wb") as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
                        received += len(chunk)
                expected = response.headers.get("Content-Length")
                if expected is not None and received != int(expected):
                    raise OSError("Download ended early")
            return
        except OSError:
            if attempt == 2:
                raise


def fetch(name, destination):
    path = destination / name
    expected = HASHES.get(name)
    if path.exists():
        if expected and file_hash(path) != expected:
            raise ValueError(f"Existing archive checksum mismatch: {path}")
        with ZipFile(path) as bundle:
            if bundle.testzip() is not None:
                raise ValueError(f"Damaged archive: {path}")
        return path
    destination.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".zip.partial")
    download(SOURCES[name], partial)
    try:
        if expected and file_hash(partial) != expected:
            raise ValueError(f"Downloaded archive checksum mismatch: {name}")
        with ZipFile(partial) as bundle:
            if bundle.testzip() is not None:
                raise ValueError(f"Damaged downloaded archive: {name}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Download UTD-MHAD from its official host. See DATA.md.")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--video", action="store_true", help="Also download the larger RGB archive")
    args = parser.parse_args()
    for name in (["Inertial.zip", "RGB.zip"] if args.video else ["Inertial.zip"]):
        path = fetch(name, args.output)
        print(f"{path}: {file_hash(path)}")


if __name__ == "__main__":
    main()
