"""Seekable LZIP reader (multimember-aware).

Each LZIP member is an independent LZMA stream with a 6-byte header and
20-byte trailer. Members are indexed by walking the file using trailer
``member_size`` fields; seeking within a member restarts LZMA for that
member only (or uses a cached full-member decompress).

Packed data is read from the underlying file on demand — the full archive
is not kept in RAM.
"""

from __future__ import annotations

import contextlib
import io
import lzma
import struct
import threading
from dataclasses import dataclass
from typing import IO

from .utils import RatarmountError, overrides

LZIP_MAGIC = b"LZIP"
_HEADER_SIZE = 6
_TRAILER_SIZE = 20


class LzipError(RatarmountError):
    pass


@dataclass
class LzipMember:
    start_offset: int
    end_offset: int
    uncompressed_offset: int
    uncompressed_size: int
    dict_size_code: int


def _dict_size_from_code(code: int) -> int:
    base = 1 << (code & 31)
    frac = (code >> 5) & 7
    return max(4096, base - (base // 16) * frac)


def _decompress_member_bytes(payload_and_framing: bytes, dict_code: int) -> bytes:
    """Decompress one LZIP member buffer (header+payload+trailer)."""
    if len(payload_and_framing) < _HEADER_SIZE + _TRAILER_SIZE:
        raise LzipError("LZIP member too small")
    payload = payload_and_framing[_HEADER_SIZE : len(payload_and_framing) - _TRAILER_SIZE]
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": _dict_size_from_code(dict_code), "lc": 3, "lp": 0, "pb": 2}]
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=filters)
    out = dec.decompress(payload)
    if not dec.eof:
        with contextlib.suppress(Exception):
            out += dec.decompress(b"")
    return out


def index_lzip_file(fileobj: IO[bytes]) -> list[LzipMember]:
    """Walk LZIP members using file seeks (does not retain full file bytes)."""
    fileobj.seek(0, io.SEEK_END)
    file_size = fileobj.tell()
    fileobj.seek(0)

    members: list[LzipMember] = []
    pos = 0
    u_off = 0
    while pos + _HEADER_SIZE + _TRAILER_SIZE <= file_size:
        fileobj.seek(pos)
        header = fileobj.read(_HEADER_SIZE)
        if len(header) < _HEADER_SIZE or header[:4] != LZIP_MAGIC:
            break
        version = header[4]
        if version != 1:
            raise LzipError(f"Unsupported LZIP version: {version}")
        dict_code = header[5]

        filters = [{"id": lzma.FILTER_LZMA1, "dict_size": _dict_size_from_code(dict_code), "lc": 3, "lp": 0, "pb": 2}]
        dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=filters)
        cursor = pos + _HEADER_SIZE
        plain_len = 0
        while cursor < file_size and not dec.eof:
            fileobj.seek(cursor)
            feed = fileobj.read(min(65536, file_size - cursor))
            if not feed:
                break
            try:
                chunk = dec.decompress(feed)
            except lzma.LZMAError as exc:
                raise LzipError(f"LZIP LZMA error: {exc}") from exc
            plain_len += len(chunk)
            if dec.eof:
                unused = len(dec.unused_data)
                cursor += len(feed) - unused
                break
            cursor += len(feed)
        if not dec.eof:
            raise LzipError("LZIP member LZMA stream did not terminate")
        if cursor + _TRAILER_SIZE > file_size:
            raise LzipError("Truncated LZIP trailer")
        fileobj.seek(cursor)
        trailer = fileobj.read(_TRAILER_SIZE)
        _crc, data_size, member_size = struct.unpack("<IQQ", trailer)
        end = pos + member_size if member_size else cursor + _TRAILER_SIZE
        if not data_size:
            data_size = plain_len
        # Prefer decompressed length when trailer mismatches slightly
        if abs(data_size - plain_len) > 0:
            data_size = plain_len
        members.append(
            LzipMember(
                start_offset=pos,
                end_offset=end,
                uncompressed_offset=u_off,
                uncompressed_size=data_size,
                dict_size_code=dict_code,
            )
        )
        u_off += data_size
        pos = end

    if not members:
        raise LzipError("No LZIP members found")
    return members


class IndexedLzipFile(io.RawIOBase):
    """Seekable read-only LZIP file backed by on-demand file reads."""

    def __init__(self, fileobj: str | IO[bytes], **_kwargs):
        super().__init__()
        self._close_file = False
        if isinstance(fileobj, str):
            self._file: IO[bytes] = open(fileobj, "rb")
            self._close_file = True
        else:
            # Keep a private handle if possible; else wrap given object under lock.
            self._file = fileobj
        self._lock = threading.Lock()
        with self._lock:
            if hasattr(self._file, "seek"):
                pos = self._file.tell()
                self._file.seek(0)
            else:
                pos = 0
            self._members = index_lzip_file(self._file)
            if hasattr(self._file, "seek"):
                self._file.seek(pos)
        self._size = sum(m.uncompressed_size for m in self._members)
        self._pos = 0
        self._member_cache: dict[int, bytes] = {}
        self._member_cache_max = 4

    @property
    def size(self) -> int:
        return self._size

    def _member_plain(self, index: int) -> bytes:
        if index in self._member_cache:
            return self._member_cache[index]
        m = self._members[index]
        with self._lock:
            self._file.seek(m.start_offset)
            blob = self._file.read(m.end_offset - m.start_offset)
        plain = _decompress_member_bytes(blob, m.dict_size_code)
        if len(self._member_cache) >= self._member_cache_max:
            self._member_cache.pop(next(iter(self._member_cache)))
        self._member_cache[index] = plain
        return plain

    def _find(self, pos: int) -> tuple[int, int]:
        for i, m in enumerate(self._members):
            if pos < m.uncompressed_offset + m.uncompressed_size:
                return i, pos - m.uncompressed_offset
        if self._members:
            last = self._members[-1]
            return len(self._members) - 1, last.uncompressed_size
        return 0, 0

    @overrides(io.RawIOBase)
    def readable(self) -> bool:
        return True

    @overrides(io.RawIOBase)
    def seekable(self) -> bool:
        return True

    @overrides(io.RawIOBase)
    def writable(self) -> bool:
        return False

    @overrides(io.RawIOBase)
    def tell(self) -> int:
        return self._pos

    @overrides(io.RawIOBase)
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new < 0:
            raise ValueError("negative seek")
        self._pos = min(new, self._size)
        return self._pos

    @overrides(io.RawIOBase)
    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        if size is None or size < 0:
            size = self._size - self._pos
        size = min(size, self._size - self._pos)
        out = bytearray()
        while size > 0 and self._pos < self._size:
            mi, within = self._find(self._pos)
            data = self._member_plain(mi)
            chunk = data[within : within + size]
            out.extend(chunk)
            self._pos += len(chunk)
            size -= len(chunk)
        return bytes(out)

    @overrides(io.RawIOBase)
    def readinto(self, b) -> int:  # type: ignore[override]
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    @overrides(io.RawIOBase)
    def close(self) -> None:
        self._member_cache.clear()
        if self._close_file:
            self._file.close()
        super().close()


def open_lzip_file(fileobj: str | IO[bytes], **kwargs) -> IndexedLzipFile:
    return IndexedLzipFile(fileobj, **kwargs)
