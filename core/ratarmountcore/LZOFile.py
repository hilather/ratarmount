"""Seekable LZOP (.lzo) reader with block index for random access.

LZOP file layout (big-endian fields after magic):
https://www.lzop.org/lzop_format.php (historical) / lzop source lzop.h

Each block stores uncompressed size, compressed size, optional checksums, and
payload. Blocks are independent → true random access by decompressing only
the target block(s).
"""

from __future__ import annotations

import ctypes
import io
import logging
import struct
from dataclasses import dataclass
from typing import IO, Optional, Union

from .utils import RatarmountError, overrides

logger = logging.getLogger(__name__)

LZOP_MAGIC = bytes([0x89, 0x4C, 0x5A, 0x4F, 0x00, 0x0D, 0x0A, 0x1A, 0x0A])

F_ADLER32_D = 0x00000001
F_ADLER32_C = 0x00000002
F_CRC32_D = 0x00000100
F_CRC32_C = 0x00000200
F_H_EXTRA_FIELD = 0x00000040
F_H_CRC32 = 0x00001000
F_H_FILTER = 0x00000800

_lzo_lib = None


class LZOError(RatarmountError):
    pass


def _load_lzo():
    global _lzo_lib
    if _lzo_lib is not None:
        return _lzo_lib
    for name in ("liblzo2.so.2", "liblzo2.so", "liblzo2.dylib", "lzo2.dll"):
        try:
            lib = ctypes.CDLL(name)
            lib.lzo1x_decompress_safe.argtypes = [
                ctypes.c_char_p,
                ctypes.c_uint,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint),
            ]
            lib.lzo1x_decompress_safe.restype = ctypes.c_int
            _lzo_lib = lib
            return lib
        except OSError:
            continue
    raise LZOError(
        "liblzo2 is required for LZOP support. Install the system package (e.g. liblzo2-2) "
        "or ensure liblzo2 is on the library path."
    )


def _read_exact(fileobj: IO[bytes], n: int) -> bytes:
    data = fileobj.read(n)
    if len(data) != n:
        raise LZOError(f"Unexpected EOF (wanted {n}, got {len(data)})")
    return data


def lzo_decompress_block(src: bytes, uncompressed_size: int) -> bytes:
    lib = _load_lzo()
    dst = ctypes.create_string_buffer(uncompressed_size)
    dst_len = ctypes.c_uint(uncompressed_size)
    rc = lib.lzo1x_decompress_safe(src, len(src), dst, ctypes.byref(dst_len))
    if rc != 0:
        raise LZOError(f"lzo1x_decompress_safe failed with code {rc}")
    return dst.raw[: dst_len.value]


@dataclass
class LZOBlockInfo:
    data_offset: int
    compressed_size: int
    uncompressed_offset: int
    uncompressed_size: int
    is_stored: bool


@dataclass
class LZOFileInfo:
    flags: int
    blocks: list[LZOBlockInfo]
    total_uncompressed: int
    header_end: int


def parse_lzop_file(fileobj: IO[bytes]) -> LZOFileInfo:
    start = fileobj.tell()
    magic = _read_exact(fileobj, 9)
    if magic != LZOP_MAGIC:
        raise LZOError(f"Invalid LZOP magic: {magic!r}")

    version, _lib_version, _version_needed = struct.unpack(">HHH", _read_exact(fileobj, 6))
    method = _read_exact(fileobj, 1)[0]
    _level = _read_exact(fileobj, 1)[0]
    (flags,) = struct.unpack(">I", _read_exact(fileobj, 4))

    if flags & F_H_FILTER:
        _read_exact(fileobj, 4)  # filter id

    _mode, _mtime_low, _mtime_high = struct.unpack(">III", _read_exact(fileobj, 12))
    name_len = _read_exact(fileobj, 1)[0]
    if name_len:
        _read_exact(fileobj, name_len)

    # Header checksum (Adler-32 or CRC-32 of header bytes after magic).
    _read_exact(fileobj, 4)

    if flags & F_H_EXTRA_FIELD:
        (extra_len,) = struct.unpack(">I", _read_exact(fileobj, 4))
        _read_exact(fileobj, extra_len)
        _read_exact(fileobj, 4)  # extra checksum

    # Method 1/2/3 are LZO1X variants we handle via lzo1x_decompress_safe.
    if method not in (1, 2, 3):
        raise LZOError(f"Unsupported LZOP method: {method}")

    header_end = fileobj.tell()
    blocks: list[LZOBlockInfo] = []
    u_off = 0

    while True:
        (usize,) = struct.unpack(">I", _read_exact(fileobj, 4))
        if usize == 0:
            break
        (csize,) = struct.unpack(">I", _read_exact(fileobj, 4))
        if flags & (F_ADLER32_D | F_CRC32_D):
            _read_exact(fileobj, 4)
        if csize < usize and (flags & (F_ADLER32_C | F_CRC32_C)):
            _read_exact(fileobj, 4)
        data_offset = fileobj.tell()
        _read_exact(fileobj, csize)
        is_stored = csize == usize
        blocks.append(
            LZOBlockInfo(
                data_offset=data_offset,
                compressed_size=csize,
                uncompressed_offset=u_off,
                uncompressed_size=usize,
                is_stored=is_stored,
            )
        )
        u_off += usize

    return LZOFileInfo(flags=flags, blocks=blocks, total_uncompressed=u_off, header_end=header_end)


class IndexedLZOFile(io.RawIOBase):
    """Seekable read-only LZOP file."""

    def __init__(self, fileobj: Union[str, IO[bytes]], **_kwargs):
        super().__init__()
        _load_lzo()
        self._close_file = False
        if isinstance(fileobj, str):
            self._file: IO[bytes] = open(fileobj, "rb")
            self._close_file = True
        else:
            self._file = fileobj
            self._file.seek(0)
        self._info = parse_lzop_file(self._file)
        self._size = self._info.total_uncompressed
        self._pos = 0
        self._cache: Optional[tuple[int, bytes]] = None  # block index, data

    @property
    def size(self) -> int:
        return self._size

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

    def _decompress_block(self, index: int) -> bytes:
        if self._cache and self._cache[0] == index:
            return self._cache[1]
        block = self._info.blocks[index]
        self._file.seek(block.data_offset)
        payload = _read_exact(self._file, block.compressed_size)
        if block.is_stored:
            data = payload
        else:
            data = lzo_decompress_block(payload, block.uncompressed_size)
        self._cache = (index, data)
        return data

    def _find_block(self, pos: int) -> tuple[int, int]:
        for i, block in enumerate(self._info.blocks):
            if pos < block.uncompressed_offset + block.uncompressed_size:
                return i, pos - block.uncompressed_offset
        if self._info.blocks:
            last = self._info.blocks[-1]
            return len(self._info.blocks) - 1, last.uncompressed_size
        return 0, 0

    @overrides(io.RawIOBase)
    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        if size is None or size < 0:
            size = self._size - self._pos
        size = min(size, self._size - self._pos)
        out = bytearray()
        while size > 0 and self._pos < self._size:
            bi, within = self._find_block(self._pos)
            data = self._decompress_block(bi)
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
        if self._close_file and self._file is not None:
            self._file.close()
        super().close()


def open_lzo_file(fileobj: Union[str, IO[bytes]], **kwargs) -> IndexedLZOFile:
    return IndexedLZOFile(fileobj, **kwargs)
