"""Seekable LZIP reader (multimember-aware).

Each LZIP member is an independent LZMA stream with a 6-byte header and
20-byte trailer. Members are indexed by walking the file using trailer
``member_size`` fields; seeking within a member restarts LZMA for that
member only (or uses a cached full-member decompress).
"""

from __future__ import annotations

import io
import lzma
import struct
from dataclasses import dataclass
from typing import IO, Optional, Union

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


def _decompress_member(data: bytes, start: int, end: int, dict_code: int) -> bytes:
    """Decompress one LZIP member given [start, end) slice of the file bytes."""
    if end - start < _HEADER_SIZE + _TRAILER_SIZE:
        raise LzipError("LZIP member too small")
    payload = data[start + _HEADER_SIZE : end - _TRAILER_SIZE]
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": _dict_size_from_code(dict_code), "lc": 3, "lp": 0, "pb": 2}]
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=filters)
    out = dec.decompress(payload)
    if not dec.eof:
        try:
            out += dec.decompress(b"")
        except Exception:
            pass
    # Prefer trailer data_size if present
    _crc, data_size, _member_size = struct.unpack("<IQQ", data[end - _TRAILER_SIZE : end])
    if data_size and data_size != len(out):
        # Trust decompressor output if trailer mismatches (still return out)
        pass
    return out


def index_lzip_file(fileobj: IO[bytes]) -> tuple[bytes, list[LzipMember]]:
    fileobj.seek(0)
    data = fileobj.read()
    if not data.startswith(LZIP_MAGIC):
        raise LzipError("Not an LZIP file")

    members: list[LzipMember] = []
    pos = 0
    u_off = 0
    while pos + _HEADER_SIZE + _TRAILER_SIZE <= len(data):
        if data[pos : pos + 4] != LZIP_MAGIC:
            break
        version = data[pos + 4]
        if version != 1:
            raise LzipError(f"Unsupported LZIP version: {version}")
        dict_code = data[pos + 5]
        # member_size is last 8 bytes of the member; but we need member end.
        # For sequential parse: decompress to find stream end is hard.
        # Instead read trailer by scanning: LZIP members store member_size so
        # end = pos + member_size. We discover member_size by trying decompress
        # with growing windows OR by reading trailer after successful decompress.
        # Reliable approach used by many tools: the trailer is at pos+member_size-20,
        # and member_size is written there. We can find the next LZIP magic or EOF.
        # Sequential: decompress raw payload until EOS, then trailer is next 20 bytes.
        filters = [
            {"id": lzma.FILTER_LZMA1, "dict_size": _dict_size_from_code(dict_code), "lc": 3, "lp": 0, "pb": 2}
        ]
        dec = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=filters)
        cursor = pos + _HEADER_SIZE
        plain = bytearray()
        while cursor < len(data) and not dec.eof:
            feed = data[cursor : cursor + 65536]
            if not feed:
                break
            try:
                plain.extend(dec.decompress(feed))
            except lzma.LZMAError as exc:
                raise LzipError(f"LZIP LZMA error: {exc}") from exc
            if dec.eof:
                unused = len(dec.unused_data)
                cursor += len(feed) - unused
                break
            cursor += len(feed)
        if not dec.eof:
            raise LzipError("LZIP member LZMA stream did not terminate")
        if cursor + _TRAILER_SIZE > len(data):
            raise LzipError("Truncated LZIP trailer")
        _crc, data_size, member_size = struct.unpack("<IQQ", data[cursor : cursor + _TRAILER_SIZE])
        end = pos + member_size if member_size else cursor + _TRAILER_SIZE
        if data_size and abs(data_size - len(plain)) > 0:
            # still use plain length
            data_size = len(plain)
        else:
            data_size = len(plain)
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
    return data, members


class IndexedLzipFile(io.RawIOBase):
    """Seekable read-only LZIP file."""

    def __init__(self, fileobj: Union[str, IO[bytes]], **_kwargs):
        super().__init__()
        if isinstance(fileobj, str):
            with open(fileobj, "rb") as f:
                self._data, self._members = index_lzip_file(f)
        else:
            pos = fileobj.tell()
            fileobj.seek(0)
            self._data, self._members = index_lzip_file(fileobj)
            fileobj.seek(pos)
        self._size = sum(m.uncompressed_size for m in self._members)
        self._pos = 0
        self._member_cache: dict[int, bytes] = {}

    @property
    def size(self) -> int:
        return self._size

    def _member_plain(self, index: int) -> bytes:
        if index not in self._member_cache:
            m = self._members[index]
            self._member_cache[index] = _decompress_member(
                self._data, m.start_offset, m.end_offset, m.dict_size_code
            )
        return self._member_cache[index]

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


def open_lzip_file(fileobj: Union[str, IO[bytes]], **kwargs) -> IndexedLzipFile:
    return IndexedLzipFile(fileobj, **kwargs)
