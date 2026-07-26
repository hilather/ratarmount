"""ISO 9660 MountSource with random access via extent LBAs."""

from __future__ import annotations

import stat
import struct
from typing import IO, TYPE_CHECKING, Optional, Union

from ratarmountcore.mountsource.formats.stenciled import StenciledArchiveMountSource, make_file_row
from ratarmountcore.SQLiteIndex import SQLiteIndex
from ratarmountcore.utils import RatarmountError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

SECTOR = 2048
PVD_OFFSET = 16 * SECTOR  # primary volume descriptor starts at sector 16


def _read_both_endian_u32(data: bytes, offset: int) -> int:
    # ISO 9660 both-endian: little then big; use little.
    return struct.unpack_from("<I", data, offset)[0]


def _parse_directory_record(data: bytes, offset: int) -> dict | None:
    if offset >= len(data):
        return None
    length = data[offset]
    if length == 0:
        return None
    if offset + length > len(data):
        raise RatarmountError("Truncated ISO directory record")
    rec = data[offset : offset + length]
    extent = _read_both_endian_u32(rec, 2)
    size = _read_both_endian_u32(rec, 10)
    flags = rec[25]
    name_len = rec[32]
    name_bytes = rec[33 : 33 + name_len]
    # 0x00 = current dir, 0x01 = parent dir
    if name_bytes in (b"\x00", b"\x01"):
        name = None
    else:
        # Strip ISO version suffix ";1"
        name = name_bytes.split(b";", 1)[0].decode("ascii", errors="replace").rstrip(".")
    return {
        "length": length,
        "extent": extent,
        "size": size,
        "flags": flags,
        "name": name,
        "is_dir": bool(flags & 0x02),
    }


def _read_sector(fileobj: IO[bytes], sector: int) -> bytes:
    fileobj.seek(sector * SECTOR)
    data = fileobj.read(SECTOR)
    if len(data) < SECTOR:
        raise RatarmountError(f"Short read at ISO sector {sector}")
    return data


def _walk_directory(
    fileobj: IO[bytes],
    extent: int,
    size: int,
    path_prefix: str,
    rows: list[tuple],
    seen: set[int],
) -> None:
    if extent in seen:
        return
    seen.add(extent)

    remaining = size
    sector = extent
    while remaining > 0:
        data = _read_sector(fileobj, sector)
        to_parse = min(SECTOR, remaining)
        offset = 0
        while offset < to_parse:
            if data[offset] == 0:
                # padding to end of sector
                break
            rec = _parse_directory_record(data, offset)
            if rec is None:
                break
            offset += rec["length"]
            if rec["name"] is None:
                continue
            full = f"{path_prefix}/{rec['name']}" if path_prefix else rec["name"]
            full = full.lstrip("/")
            if rec["is_dir"]:
                path, name = SQLiteIndex.normpath(full).rsplit("/", 1)
                rows.append(
                    make_file_row(
                        path=path,
                        name=name,
                        header_offset=rec["extent"] * SECTOR,
                        data_offset=rec["extent"] * SECTOR,
                        size=0,
                        mtime=0,
                        mode=0o755 | stat.S_IFDIR,
                    )
                )
                _walk_directory(fileobj, rec["extent"], rec["size"], full, rows, seen)
            else:
                path, name = SQLiteIndex.normpath(full).rsplit("/", 1)
                rows.append(
                    make_file_row(
                        path=path,
                        name=name,
                        header_offset=rec["extent"] * SECTOR,
                        data_offset=rec["extent"] * SECTOR,
                        size=rec["size"],
                        mtime=0,
                        mode=0o644 | stat.S_IFREG,
                    )
                )
        remaining -= SECTOR
        sector += 1


def parse_iso9660_archive(fileobj: IO[bytes]) -> list[tuple]:
    """Parse ISO9660 primary volume descriptor and directory tree."""
    fileobj.seek(PVD_OFFSET)
    pvd = fileobj.read(SECTOR)
    if len(pvd) < SECTOR or pvd[1:6] != b"CD001" or pvd[0] != 1:
        # Try UDF is out of scope; require ISO9660 PVD.
        raise RatarmountError("Not a valid ISO 9660 image (missing primary volume descriptor)")

    # Root directory record at offset 156 in PVD (34 bytes).
    root = _parse_directory_record(pvd, 156)
    if root is None:
        raise RatarmountError("ISO 9660 primary volume descriptor has no root directory")

    rows: list[tuple] = []
    seen: set[int] = set()
    _walk_directory(fileobj, root["extent"], root["size"], "", rows, seen)
    if not rows:
        # Empty ISO still valid.
        pass
    return rows


class ISO9660MountSource(StenciledArchiveMountSource):
    def __init__(self, fileOrPath: str | IO[bytes] | Path, **options) -> None:
        def build_rows(fileobj: IO[bytes]) -> Iterable[tuple]:
            fileobj.seek(0)
            return parse_iso9660_archive(fileobj)

        super().__init__(fileOrPath, backendName="ISO9660MountSource", build_rows=build_rows, **options)
