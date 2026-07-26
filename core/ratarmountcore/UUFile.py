"""Seekable reader for Unix uuencode streams (classic and begin-base64).

Uuencode has no internal seek points. This reader decodes once and serves a
seekable view of the payload — same model as compress (.Z), preferred over
libarchive re-scan on every open.
"""

from __future__ import annotations

import binascii
import base64
import io
import re
from typing import IO, Union

from .utils import RatarmountError, overrides


class UUError(RatarmountError):
    pass


_BEGIN_RE = re.compile(rb"^begin(?:-base64)?\s+(\d+)\s+(\S+)\s*$", re.MULTILINE)
_END_MARKERS = (b"end", b"====")


def decode_uu(data: bytes) -> tuple[bytes, str, int]:
    """Decode a uuencode (or begin-base64) buffer.

    Returns (payload, filename, mode).
    """
    if not data.lstrip().startswith(b"begin"):
        raise UUError("Not a uuencode stream (missing begin line)")

    # Normalize newlines
    text = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    match = _BEGIN_RE.search(text)
    if not match:
        raise UUError("Invalid uuencode begin line")
    mode = int(match.group(1), 8) if match.group(1) else 0o644
    name = match.group(2).decode("latin-1", errors="replace")
    is_base64 = text[match.start() : match.start() + 12].startswith(b"begin-base64")
    body = text[match.end() :]
    if body.startswith(b"\n"):
        body = body[1:]

    if is_base64:
        lines = []
        for line in body.split(b"\n"):
            line = line.strip()
            if not line or line in _END_MARKERS:
                break
            lines.append(line)
        try:
            payload = base64.b64decode(b"".join(lines), validate=False)
        except Exception as exc:
            raise UUError(f"base64 uuencode decode failed: {exc}") from exc
        return payload, name, mode

    # Classic uuencode via binascii
    # Re-build a minimal stream for a2b_uu: begin line + body through end
    reconstructed = text[match.start() :]
    # Ensure trailing end
    if b"\nend" not in reconstructed and not reconstructed.rstrip().endswith(b"end"):
        reconstructed = reconstructed.rstrip() + b"\nend\n"
    try:
        payload = binascii.a2b_uu(reconstructed)
    except binascii.Error:
        # a2b_uu expects pure body lines without begin/end on some Python versions;
        # decode line-by-line.
        out = bytearray()
        for line in body.split(b"\n"):
            if not line or line.strip() in _END_MARKERS:
                break
            # Length byte is first char; tolerate short lines
            try:
                out.extend(binascii.a2b_uu(line + b"\n"))
            except binascii.Error:
                try:
                    out.extend(binascii.a2b_uu(line))
                except binascii.Error as exc:
                    raise UUError(f"uuencode line decode failed: {exc}") from exc
        payload = bytes(out)
    return payload, name, mode


class IndexedUUFile(io.RawIOBase):
    """Seekable read-only view of a uuencode file payload."""

    def __init__(self, fileobj: Union[str, IO[bytes]], **_kwargs):
        super().__init__()
        if isinstance(fileobj, str):
            raw = open(fileobj, "rb").read()
        else:
            pos = fileobj.tell()
            fileobj.seek(0)
            raw = fileobj.read()
            fileobj.seek(pos)

        payload, self.filename, self.mode = decode_uu(raw)
        self._buffer = io.BytesIO(payload)
        self._size = len(payload)

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


def open_uu_file(fileobj: Union[str, IO[bytes]], **kwargs) -> IndexedUUFile:
    return IndexedUUFile(fileobj, **kwargs)
