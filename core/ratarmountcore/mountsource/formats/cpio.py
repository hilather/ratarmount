"""CPIO archive MountSource with true random access via member offsets."""

from __future__ import annotations

import os
import stat
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Optional, Union

from ratarmountcore.mountsource.formats.stenciled import StenciledArchiveMountSource, make_file_row
from ratarmountcore.SQLiteIndex import SQLiteIndex
from ratarmountcore.utils import RatarmountError

# newc/crc ASCII header is 110 bytes of hex digits after the 6-byte magic.
_NEWC_HEADER_SIZE = 110
_ODC_HEADER_SIZE = 76
_BIN_HEADER_SIZE = 26


def _align(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) // alignment * alignment


def _parse_newc_like(fileobj: IO[bytes], encoding: str = "utf-8") -> list[tuple]:
    """Parse SVR4 newc (070701) or crc (070702) CPIO archives."""
    rows: list[tuple] = []
    while True:
        header_offset = fileobj.tell()
        magic = fileobj.read(6)
        if len(magic) < 6:
            break
        if magic not in (b"070701", b"070702"):
            # Allow trailing padding zeros after TRAILER.
            if magic == b"\x00\x00\x00\x00\x00\x00" or set(magic) <= {0}:
                break
            raise RatarmountError(f"Invalid newc/crc CPIO magic at {header_offset}: {magic!r}")

        rest = fileobj.read(_NEWC_HEADER_SIZE - 6)
        if len(rest) < _NEWC_HEADER_SIZE - 6:
            raise RatarmountError("Truncated newc/crc CPIO header")
        fields = rest.decode("ascii")
        # c_ino, c_mode, c_uid, c_gid, c_nlink, c_mtime, c_filesize,
        # c_devmajor, c_devminor, c_rdevmajor, c_rdevminor, c_namesize, c_check
        try:
            values = [int(fields[i : i + 8], 16) for i in range(0, 13 * 8, 8)]
        except ValueError as exc:
            raise RatarmountError("Invalid hex in newc CPIO header") from exc
        mode, uid, gid, _nlink, mtime, filesize = values[1], values[2], values[3], values[4], values[5], values[6]
        namesize = values[11]

        name_bytes = fileobj.read(namesize)
        if len(name_bytes) < namesize:
            raise RatarmountError("Truncated CPIO filename")
        # Name includes trailing NUL; strip it and any padding to 4-byte boundary after header+name.
        name = name_bytes.split(b"\x00", 1)[0].decode(encoding, errors="surrogateescape")

        # Header is 110 bytes; name field is namesize bytes; total padded to multiple of 4.
        name_end = header_offset + _NEWC_HEADER_SIZE + namesize
        data_offset = _align(name_end, 4)
        fileobj.seek(data_offset)

        if name == "TRAILER!!!":
            break

        data = fileobj.read(filesize)
        if len(data) < filesize:
            raise RatarmountError(f"Truncated CPIO data for {name!r}")
        # Data padded to 4 bytes.
        fileobj.seek(data_offset + _align(filesize, 4))

        rows.append(_entry_to_row(name, mode, mtime, filesize, header_offset, data_offset, uid, gid, data))
    return rows


def _parse_odc(fileobj: IO[bytes], encoding: str = "utf-8") -> list[tuple]:
    """Parse portable ASCII odc (070707) CPIO archives."""
    rows: list[tuple] = []
    while True:
        header_offset = fileobj.tell()
        magic = fileobj.read(6)
        if len(magic) < 6:
            break
        if magic != b"070707":
            if set(magic) <= {0}:
                break
            raise RatarmountError(f"Invalid odc CPIO magic at {header_offset}: {magic!r}")

        rest = fileobj.read(_ODC_HEADER_SIZE - 6)
        if len(rest) < _ODC_HEADER_SIZE - 6:
            raise RatarmountError("Truncated odc CPIO header")
        # All fields octal ASCII. Layout after magic (6):
        # dev(6) ino(6) mode(6) uid(6) gid(6) nlink(6) rdev(6) mtime(11) namesize(6) filesize(11)
        try:
            # Portable ASCII odc field widths after 6-byte magic (total header 76):
            # dev6 ino6 mode6 uid6 gid6 nlink6 rdev6 mtime11 namesize6 filesize11
            s = rest.decode("ascii")
            mode = int(s[12:18], 8)
            uid = int(s[18:24], 8)
            gid = int(s[24:30], 8)
            mtime = int(s[42:53], 8)
            namesize = int(s[53:59], 8)
            filesize = int(s[59:70], 8)
        except ValueError as exc:
            raise RatarmountError("Invalid octal in odc CPIO header") from exc

        name_bytes = fileobj.read(namesize)
        if len(name_bytes) < namesize:
            raise RatarmountError("Truncated CPIO filename")
        name = name_bytes.split(b"\x00", 1)[0].decode(encoding, errors="surrogateescape")
        data_offset = fileobj.tell()

        if name == "TRAILER!!!":
            break

        data = fileobj.read(filesize)
        if len(data) < filesize:
            raise RatarmountError(f"Truncated CPIO data for {name!r}")

        rows.append(_entry_to_row(name, mode, mtime, filesize, header_offset, data_offset, uid, gid, data))
    return rows


def _parse_bin(fileobj: IO[bytes], encoding: str = "utf-8") -> list[tuple]:
    """Parse binary CPIO (old binary, magic 070707 as 0x71c7 little or big endian)."""
    rows: list[tuple] = []
    while True:
        header_offset = fileobj.tell()
        magic_bytes = fileobj.read(2)
        if len(magic_bytes) < 2:
            break
        # Magic 0x71c7: little-endian archives store bytes c7 71, big-endian store 71 c7.
        if magic_bytes == b"\xc7\x71":
            endian = "<"
        elif magic_bytes == b"\x71\xc7":
            endian = ">"
        else:
            if magic_bytes == b"\x00\x00":
                break
            raise RatarmountError(f"Invalid binary CPIO magic at {header_offset}: {magic_bytes!r}")

        rest = fileobj.read(24)
        if len(rest) < 24:
            raise RatarmountError("Truncated binary CPIO header")
        # Standard old binary after magic:
        #   c_dev, c_ino, c_mode, c_uid, c_gid, c_nlink, c_rdev,
        #   c_mtime[0], c_mtime[1], c_namesize, c_filesize[0], c_filesize[1]
        fields = struct.unpack(endian + "HHHHHHHHHHHH", rest)
        mode = fields[2]
        uid = fields[3]
        gid = fields[4]
        mtime = (fields[7] << 16) | fields[8]
        namesize = fields[9]
        filesize = (fields[10] << 16) | fields[11]

        # namesize includes NUL; header+name padded to even
        name_bytes = fileobj.read(namesize)
        if len(name_bytes) < namesize:
            raise RatarmountError("Truncated binary CPIO filename")
        name = name_bytes.split(b"\x00", 1)[0].decode(encoding, errors="surrogateescape")
        # Align to even after name
        name_end = fileobj.tell()
        if name_end % 2 == 1:
            fileobj.read(1)
        data_offset = fileobj.tell()

        if name == "TRAILER!!!":
            break

        data = fileobj.read(filesize)
        if len(data) < filesize:
            raise RatarmountError(f"Truncated CPIO data for {name!r}")
        if filesize % 2 == 1:
            fileobj.read(1)

        rows.append(_entry_to_row(name, mode, mtime, filesize, header_offset, data_offset, uid, gid, data))
    return rows


def _entry_to_row(
    name: str,
    mode: int,
    mtime: int,
    filesize: int,
    header_offset: int,
    data_offset: int,
    uid: int,
    gid: int,
    data: bytes,
) -> tuple:
    file_type = mode & 0o170000
    is_dir = file_type == 0o040000
    is_lnk = file_type == 0o120000
    linkname = ""
    size = filesize
    if is_lnk:
        linkname = data.decode("utf-8", errors="surrogateescape")
        size = 0
    elif is_dir:
        size = 0

    # Ensure type bits present.
    if is_dir and not stat.S_ISDIR(mode):
        mode = (mode & 0o7777) | stat.S_IFDIR
    elif is_lnk and not stat.S_ISLNK(mode):
        mode = (mode & 0o7777) | stat.S_IFLNK
    elif not is_dir and not is_lnk and not stat.S_ISREG(mode):
        mode = (mode & 0o7777) | stat.S_IFREG

    path, base = SQLiteIndex.normpath(name).rsplit("/", 1)
    return make_file_row(
        path=path,
        name=base,
        header_offset=header_offset,
        data_offset=data_offset,
        size=size,
        mtime=float(mtime),
        mode=mode,
        linkname=linkname,
        uid=uid,
        gid=gid,
    )


def parse_cpio_archive(fileobj: IO[bytes], encoding: str = "utf-8") -> list[tuple]:
    """Detect CPIO variant and return SQLite index rows."""
    pos = fileobj.tell()
    magic = fileobj.read(6)
    fileobj.seek(pos)
    if magic[:6] in (b"070701", b"070702"):
        return _parse_newc_like(fileobj, encoding=encoding)
    if magic[:6] == b"070707":
        return _parse_odc(fileobj, encoding=encoding)
    if magic[:2] in (b"\x71\xc7", b"\xc7\x71"):
        return _parse_bin(fileobj, encoding=encoding)
    raise RatarmountError(f"Unrecognized CPIO magic: {magic!r}")


class CPIOMountSource(StenciledArchiveMountSource):
    def __init__(self, fileOrPath: Union[str, IO[bytes], Path], **options) -> None:
        encoding = options.get("encoding", "utf-8")

        def build_rows(fileobj: IO[bytes]) -> Iterable[tuple]:
            fileobj.seek(0)
            return parse_cpio_archive(fileobj, encoding=encoding)

        super().__init__(fileOrPath, backendName="CPIOMountSource", build_rows=build_rows, **options)
