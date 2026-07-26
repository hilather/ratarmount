"""Seekable reader for Unix compress (.Z) LZW streams.

The .Z format is a single LZW stream without internal seek points. This reader
decompresses once (via unlzw3) and serves a seekable view of the result. That
matches practical needs for typical .Z sizes; huge .Z files still benefit from
random access *after* the one-time decompress, unlike libarchive re-scan on
every open.
"""

from __future__ import annotations

import io
from typing import IO, Union

from .utils import RatarmountError, overrides

try:
    import unlzw3
except ImportError:  # pragma: no cover
    unlzw3 = None  # type: ignore


class CompressZError(RatarmountError):
    pass


def _require_unlzw3() -> None:
    if unlzw3 is None:
        raise CompressZError("The 'unlzw3' package is required for .Z support. Install with: pip install unlzw3")


class IndexedCompressZFile(io.RawIOBase):
    """Seekable read-only view of a Unix compress (.Z) file."""

    def __init__(self, fileobj: Union[str, IO[bytes]], **_kwargs):
        super().__init__()
        _require_unlzw3()
        self._close_file = False
        if isinstance(fileobj, str):
            raw = open(fileobj, "rb").read()
            self._close_file = True
        else:
            pos = fileobj.tell()
            fileobj.seek(0)
            raw = fileobj.read()
            fileobj.seek(pos)

        if len(raw) < 3 or raw[0:2] not in (b"\x1f\x9d", b"\x1f\xa0"):
            raise CompressZError("Not a Unix compress (.Z) file")

        # unlzw3 accepts path or bytes depending on version
        try:
            plain = unlzw3.unlzw(raw)
        except TypeError:
            # Some versions expect a Path-like and read themselves — use temp buffer API
            plain = bytes(unlzw3.unlzw(io.BytesIO(raw)))  # type: ignore[arg-type]
        if not isinstance(plain, (bytes, bytearray)):
            plain = bytes(plain)
        self._buffer = io.BytesIO(plain)
        self._size = len(plain)

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
        return self._buffer.tell()

    @overrides(io.RawIOBase)
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._buffer.seek(offset, whence)

    @overrides(io.RawIOBase)
    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    @overrides(io.RawIOBase)
    def readinto(self, b) -> int:  # type: ignore[override]
        return self._buffer.readinto(b)  # type: ignore[arg-type]

    @overrides(io.RawIOBase)
    def close(self) -> None:
        self._buffer.close()
        super().close()


def open_compress_z_file(fileobj: Union[str, IO[bytes]], **kwargs) -> IndexedCompressZFile:
    return IndexedCompressZFile(fileobj, **kwargs)
