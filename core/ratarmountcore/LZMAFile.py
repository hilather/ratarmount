"""Seekable reader for raw .lzma (LZMA_Alone) streams.

.lzma has no internal seek points. This caches the full decompressed stream
after one pass (acceptable for typical .lzma sizes used in archives).
"""

from __future__ import annotations

import io
import lzma
from typing import IO, Union

from .utils import RatarmountError, overrides


class LZMAFileError(RatarmountError):
    pass


class IndexedLZMAFile(io.RawIOBase):
    def __init__(self, fileobj: Union[str, IO[bytes]], **_kwargs):
        super().__init__()
        if isinstance(fileobj, str):
            with open(fileobj, "rb") as f:
                raw = f.read()
        else:
            pos = fileobj.tell()
            fileobj.seek(0)
            raw = fileobj.read()
            fileobj.seek(pos)
        if len(raw) < 13:
            raise LZMAFileError("Truncated .lzma file")
        try:
            plain = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
        except lzma.LZMAError as exc:
            raise LZMAFileError(f"Failed to decompress .lzma: {exc}") from exc
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


def open_lzma_file(fileobj: Union[str, IO[bytes]], **kwargs) -> IndexedLZMAFile:
    return IndexedLZMAFile(fileobj, **kwargs)
