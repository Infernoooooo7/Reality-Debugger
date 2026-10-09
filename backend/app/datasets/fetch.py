"""Reproducible dataset acquisition.

* Downloads resume (HTTP Range) and are verified by SHA-256. The first
  download of an archive records its checksum in data/manifests/<id>.json;
  later downloads must match it or are deleted.
* Extraction is path-safe (no absolute paths, no ``..``, tar 'data' filter).
* ``remote-zip`` archives are read through HTTP range requests, so only the
  selected members of a very large zip are transferred.
* Storage is checked before downloading.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
import time
import urllib.request
import zipfile
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from app.datasets import paths
from app.datasets.sources import Archive, DatasetSource

CHUNK = 1 << 20
USER_AGENT = "reality-debugger-datasets/1.0"


class FetchError(RuntimeError):
    pass


# -- manifest -------------------------------------------------------------------------------------


def manifest_path(dataset_id: str) -> Path:
    return paths.manifests_dir() / f"{dataset_id}.json"


def load_manifest(dataset_id: str) -> dict[str, Any]:
    path = manifest_path(dataset_id)
    return json.loads(path.read_text()) if path.exists() else {}


def save_manifest(dataset_id: str, manifest: dict[str, Any]) -> None:
    path = manifest_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")


def base_manifest(src: DatasetSource) -> dict[str, Any]:
    return {
        "id": src.id,
        "name": src.name,
        "version": src.version,
        "homepage": src.homepage,
        "paper": src.paper,
        "tasks": list(src.tasks),
        "licence": {
            "annotations": src.licence.annotations,
            "images": src.licence.images,
            "commercial_use": src.licence.commercial_use,
            "verified": src.licence.verified,
            "note": src.licence.note,
        },
        "acquisition_note": src.acquisition_note,
        "adapter": src.adapter,
        "archives": {},
        "storage": "data/raw/" + src.id + " (git-ignored; RD_DATA_DIR overrides the data root)",
    }


# -- HTTP -----------------------------------------------------------------------------------------


def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})


def remote_size(url: str) -> int | None:
    with urllib.request.urlopen(_request(url, {"Range": "bytes=0-0"}), timeout=60) as resp:  # noqa: S310
        rng = resp.headers.get("Content-Range")
        if rng and "/" in rng:
            return int(rng.rsplit("/", 1)[1])
        length = resp.headers.get("Content-Length")
        return int(length) if length else None


def download(url: str, dest: Path, *, attempts: int = 4, progress: bool = True) -> None:
    """Download to ``dest`` (resuming a ``.part`` file)."""
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with urllib.request.urlopen(_request(url, headers), timeout=120) as resp:  # noqa: S310
                if have and resp.status != 206:
                    have = 0  # server ignored the range: start over
                total = resp.headers.get("Content-Length")
                total_bytes = (int(total) + have) if total else None
                started = time.monotonic()
                last = 0.0
                with part.open("ab" if have else "wb") as out:
                    done = have
                    while chunk := resp.read(CHUNK):
                        out.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if progress and now - last > 15:
                            rate = (done - have) / max(1e-6, now - started) / 1e6
                            pct = f"{100 * done / total_bytes:.0f}%" if total_bytes else f"{done / 1e6:.0f} MB"
                            print(f"    {dest.name}: {pct} ({rate:.1f} MB/s)", flush=True)
                            last = now
            part.replace(dest)
            return
        except OSError as error:
            if attempt == attempts:
                raise FetchError(f"download failed after {attempts} attempts: {url}: {error}") from error
            print(f"    {error}; retrying ({attempt}/{attempts})", flush=True)
            time.sleep(2**attempt)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (block-cached).

    Lets :mod:`zipfile` read the central directory and selected members of a
    remote archive without downloading the whole file.
    """

    def __init__(self, url: str, block: int = 4 << 20, cache_blocks: int = 16) -> None:
        super().__init__()
        self.url = url
        size = remote_size(url)
        if size is None:
            raise FetchError(f"server did not report a size for {url}")
        self.size = size
        self.block = block
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.cache_blocks = cache_blocks
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _block(self, index: int) -> bytes:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block
        end = min(self.size, start + self.block) - 1
        for attempt in range(1, 5):
            try:
                with urllib.request.urlopen(_request(self.url, {"Range": f"bytes={start}-{end}"}), timeout=120) as resp:  # noqa: S310
                    data = resp.read()
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        self.bytes_fetched += len(data)
        self.cache[index] = data
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            index, offset = divmod(self.pos, self.block)
            data = self._block(index)[offset : offset + n]
            if not data:
                break
            out += data
            self.pos += len(data)
            n -= len(data)
        return bytes(out)

    def readinto(self, b: Any) -> int:  # pragma: no cover - io plumbing
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


# -- extraction -----------------------------------------------------------------------------------


def _safe_member(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and not name.startswith("\\")


def extract_zip(archive: Path | io.RawIOBase, target: Path, prefixes: tuple[str, ...] = ()) -> int:
    count = 0
    with zipfile.ZipFile(archive) as zf:  # type: ignore[arg-type]
        for info in zf.infolist():
            if prefixes and not any(info.filename.startswith(p) for p in prefixes):
                continue
            if not _safe_member(info.filename):
                raise FetchError(f"unsafe path in archive: {info.filename}")
            out = target / info.filename
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and out.stat().st_size == info.file_size:
                count += 1
                continue
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            count += 1
    return count


def extract_tar(archive: Path, target: Path) -> int:
    count = 0
    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers() if _safe_member(m.name)]
        tf.extractall(target, members=members, filter="data")
        count = sum(1 for m in members if m.isfile())
    return count


# -- orchestration --------------------------------------------------------------------------------


def check_storage(src: DatasetSource) -> None:
    root = paths.data_root()
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    # Archive + extracted copy, with margin.
    need = int(src.approx_bytes * 2.2) if any(a.extract in ("zip", "tar") for a in src.archives) else src.approx_bytes
    if free < need:
        raise FetchError(f"not enough free space under {root}: need ~{need / 1e9:.1f} GB, have {free / 1e9:.1f} GB")


def fetch(src: DatasetSource, *, force: bool = False) -> dict[str, Any]:
    paths.ensure_layout()
    check_storage(src)
    target = paths.raw_dir(src.id)
    target.mkdir(parents=True, exist_ok=True)
    manifest = {**base_manifest(src), **{k: v for k, v in load_manifest(src.id).items() if k in ("archives", "contents")}}
    manifest.setdefault("archives", {})
    for archive in src.archives:
        record = manifest["archives"].get(archive.filename, {})
        print(f"  {archive.filename} <- {archive.url}", flush=True)
        if archive.extract == "remote-zip":
            record.update(_fetch_remote_zip(archive, target))
        else:
            record.update(_fetch_file(archive, target, record, force=force))
        record["url"] = archive.url
        manifest["archives"][archive.filename] = record
        save_manifest(src.id, manifest)  # keep progress even if a later archive fails
    manifest["fetched_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    save_manifest(src.id, manifest)
    return manifest


def _fetch_file(archive: Archive, target: Path, record: dict[str, Any], *, force: bool) -> dict[str, Any]:
    marker = target / f".{archive.filename}.extracted"
    if marker.exists() and not force:
        print("    already extracted", flush=True)
        return record
    dest = target / archive.filename
    if not dest.exists() or force:
        download(archive.url, dest)
    digest = sha256_file(dest)
    expected = record.get("sha256")
    if expected and expected != digest:
        dest.unlink()
        raise FetchError(f"SHA-256 mismatch for {archive.filename}: expected {expected}, got {digest}; file deleted")
    out: dict[str, Any] = {"sha256": digest, "bytes": dest.stat().st_size}
    if archive.extract == "zip":
        out["files_extracted"] = extract_zip(dest, target)
    elif archive.extract == "tar":
        out["files_extracted"] = extract_tar(dest, target)
    if archive.extract in ("zip", "tar"):
        marker.write_text(digest + "\n")
        if archive.delete_after_extract:
            dest.unlink()
            out["archive_deleted"] = True
    return out


def _fetch_remote_zip(archive: Archive, target: Path) -> dict[str, Any]:
    remote = HTTPRangeFile(archive.url)
    files = extract_zip(remote, target, archive.members)
    return {
        "bytes_total": remote.size,
        "bytes_transferred": remote.bytes_fetched,
        "members": list(archive.members),
        "files_extracted": files,
        "note": "partial extraction via HTTP range requests; member CRCs are checked by zipfile",
    }
