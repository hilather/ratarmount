"""RPM MountSource: locate payload and open via compression + CPIO.

RPM layout: Lead (96) → Signature header → Header → Payload.
Payload is typically a compressed CPIO archive (gzip/xz/zstd/lzma or none).
This backend seeks to the payload once and reuses seekable compression backends
plus CPIOMountSource — no libarchive re-scan for every member open.
"""

from __future__ import annotations

import contextlib
import io
import logging
import struct
import threading
from pathlib import Path
from typing import IO, Optional, Union, cast

from ratarmountcore.mountsource import FileInfo, MountSource
from ratarmountcore.mountsource.formats.cpio import CPIOMountSource
from ratarmountcore.StenciledFile import StenciledFile
from ratarmountcore.utils import RatarmountError, overrides

logger = logging.getLogger(__name__)

RPM_LEAD_MAGIC = b"\xed\xab\xee\xdb"
RPM_HEADER_MAGIC = b"\x8e\xad\xe8\x01"
RPM_LEAD_SIZE = 96

# Selected RPM header tags
RPMTAG_PAYLOADFORMAT = 1124
RPMTAG_PAYLOADCOMPRESSOR = 1125
RPM_STRING_TYPE = 6
RPM_STRING_ARRAY_TYPE = 8
RPM_I18NSTRING_TYPE = 9


class RPMError(RatarmountError):
    pass


def _align8(n: int) -> int:
    return (n + 7) & ~7


def _read_exact(fileobj: IO[bytes], n: int) -> bytes:
    data = fileobj.read(n)
    if len(data) != n:
        raise RPMError("Truncated RPM data")
    return data


def _parse_header_blob(fileobj: IO[bytes]) -> tuple[dict[int, object], int]:
    """Parse one RPM header section; return (tag→value, total_bytes_consumed)."""
    start = fileobj.tell()
    preamble = _read_exact(fileobj, 16)
    if preamble[:4] != RPM_HEADER_MAGIC:
        raise RPMError(f"Invalid RPM header magic: {preamble[:4]!r}")
    nindex, hsize = struct.unpack(">II", preamble[8:16])
    if nindex > 1_000_000 or hsize > 256 * 1024 * 1024:
        raise RPMError(f"Unreasonable RPM header sizes: nindex={nindex} hsize={hsize}")
    index_blob = _read_exact(fileobj, 16 * nindex)
    store = _read_exact(fileobj, hsize)
    tags: dict[int, object] = {}
    for i in range(nindex):
        tag, typ, offset, count = struct.unpack(">iiii", index_blob[i * 16 : (i + 1) * 16])
        if offset < 0 or offset > len(store):
            continue
        if typ in (RPM_STRING_TYPE, RPM_I18NSTRING_TYPE):
            end = store.find(b"\x00", offset)
            if end < 0:
                end = len(store)
            tags[tag] = store[offset:end].decode("utf-8", errors="replace")
        elif typ == RPM_STRING_ARRAY_TYPE and count > 0:
            values = []
            pos = offset
            for _ in range(count):
                end = store.find(b"\x00", pos)
                if end < 0:
                    break
                values.append(store[pos:end].decode("utf-8", errors="replace"))
                pos = end + 1
            tags[tag] = values
    consumed = fileobj.tell() - start
    return tags, consumed


def parse_rpm_payload_location(fileobj: IO[bytes]) -> tuple[int, int, str]:
    """Return (payload_offset, payload_size, compressor_name)."""
    fileobj.seek(0)
    lead = _read_exact(fileobj, RPM_LEAD_SIZE)
    if lead[:4] != RPM_LEAD_MAGIC:
        raise RPMError("Not an RPM package (bad lead magic)")

    # Signature header (padded to 8 bytes)
    _sig_tags, sig_len = _parse_header_blob(fileobj)
    sig_end = RPM_LEAD_SIZE + sig_len
    pad = _align8(sig_end) - sig_end
    if pad:
        _read_exact(fileobj, pad)

    header_tags, _hdr_len = _parse_header_blob(fileobj)
    payload_offset = fileobj.tell()

    fileobj.seek(0, io.SEEK_END)
    end = fileobj.tell()
    payload_size = end - payload_offset
    if payload_size < 0:
        raise RPMError("Invalid RPM payload offset")

    compressor = header_tags.get(RPMTAG_PAYLOADCOMPRESSOR)
    if isinstance(compressor, list):
        compressor = compressor[0] if compressor else None
    if not compressor:
        fileobj.seek(payload_offset)
        magic = fileobj.read(6)
        if magic.startswith(b"\x1f\x8b"):
            compressor = "gzip"
        elif magic.startswith(b"\xfd7zXZ"):
            compressor = "xz"
        elif magic.startswith(b"\x28\xb5\x2f\xfd"):
            compressor = "zstd"
        elif magic.startswith(b"\x5d\x00"):
            compressor = "lzma"
        elif magic.startswith(b"BZh"):
            compressor = "bzip2"
        elif magic[:6] in (b"070701", b"070702", b"070707"):
            compressor = "uncompressed"
        else:
            compressor = "gzip"
    else:
        compressor = str(compressor).lower()

    payload_format = header_tags.get(RPMTAG_PAYLOADFORMAT)
    if isinstance(payload_format, list):
        payload_format = payload_format[0] if payload_format else None
    if payload_format and str(payload_format).lower() not in ("cpio", "compressed_cpio", ""):
        logger.warning("RPM payload format %r may not be CPIO", payload_format)

    return payload_offset, payload_size, compressor


def _via_named_backend(region: IO[bytes], names: list[str]):
    """Open region with the first available named compression backend (never libarchive)."""
    import sys

    from ratarmountcore.compressions import COMPRESSION_BACKENDS

    for name in names:
        info = COMPRESSION_BACKENDS.get(name)
        if info is None:
            continue
        if info.requiredModules and not all(mod in sys.modules for mod, _ in info.requiredModules):
            # Try importing required modules so find works after first use.
            for mod, _pkg in info.requiredModules:
                if mod not in sys.modules:
                    with contextlib.suppress(Exception):
                        __import__(mod)
        if info.requiredModules and not all(mod in sys.modules for mod, _ in info.requiredModules):
            continue
        try:
            return info.open(region)
        except Exception:
            continue
    return None


def _open_payload_stream(
    fileobj: IO[bytes],
    lock: threading.Lock,
    offset: int,
    size: int,
    compressor: str,
) -> IO[bytes]:
    """Open a seekable view of the (possibly compressed) RPM payload."""
    region = cast(IO[bytes], StenciledFile([(fileobj, offset, size)], lock, bufferSize=65536))
    name = compressor.lower()
    if name in ("", "uncompressed", "none", "cpio"):
        return region

    if name in ("gzip", "gz"):
        stream = _via_named_backend(region, ["rapidgzip", "indexed_gzip"])
        if stream is not None:
            return stream
        import gzip

        return cast(IO[bytes], gzip.GzipFile(fileobj=region, mode="rb"))

    if name in ("xz", "lzma"):
        stream = _via_named_backend(region, ["xz", "lzmaffi", "lzma"] if name == "xz" else ["lzma"])
        if stream is not None:
            return stream
        import lzma

        fmt = lzma.FORMAT_XZ if name == "xz" else lzma.FORMAT_ALONE
        region.seek(0)
        return cast(IO[bytes], io.BytesIO(lzma.decompress(region.read(), format=fmt)))

    if name in ("zstd", "zstandard"):
        stream = _via_named_backend(region, ["indexed_zstd"])
        if stream is not None:
            return stream
        try:
            import zstandard

            region.seek(0)
            return cast(IO[bytes], io.BytesIO(zstandard.ZstdDecompressor().decompress(region.read())))
        except Exception as exc:
            raise RPMError(f"zstd RPM payload requires indexed_zstd or zstandard: {exc}") from exc

    if name in ("bzip2", "bz2"):
        stream = _via_named_backend(region, ["rapidgzip-bzip2"])
        if stream is not None:
            return stream
        import bz2

        region.seek(0)
        return cast(IO[bytes], io.BytesIO(bz2.decompress(region.read())))

    raise RPMError(f"Unsupported RPM payload compressor: {compressor!r}")


class RPMMountSource(MountSource):
    """Mount an RPM by exposing its CPIO payload through CPIOMountSource."""

    def __init__(self, fileOrPath: str | IO[bytes] | Path, **options) -> None:
        if isinstance(fileOrPath, Path):
            fileOrPath = str(fileOrPath)
        self.isFileObject = not isinstance(fileOrPath, str)
        self.fileObject: IO[bytes] = open(fileOrPath, "rb") if isinstance(fileOrPath, str) else fileOrPath
        self.fileObjectLock = threading.Lock()
        self._closed = False

        try:
            self.payload_offset, self.payload_size, self.compressor = parse_rpm_payload_location(self.fileObject)
            stream = _open_payload_stream(
                self.fileObject,
                self.fileObjectLock,
                self.payload_offset,
                self.payload_size,
                self.compressor,
            )
            cpio_options = dict(options)
            cpio_options.setdefault("indexFilePath", ":memory:")
            self._cpio = CPIOMountSource(stream, **cpio_options)
        except Exception:
            if not self.isFileObject:
                self.fileObject.close()
            raise

    @overrides(MountSource)
    def is_immutable(self) -> bool:
        return True

    @overrides(MountSource)
    def list(self, path: str):
        return self._cpio.list(path)

    @overrides(MountSource)
    def list_mode(self, path: str):
        return self._cpio.list_mode(path)

    @overrides(MountSource)
    def lookup(self, path: str, fileVersion: int = 0) -> FileInfo | None:
        return self._cpio.lookup(path, fileVersion=fileVersion)

    @overrides(MountSource)
    def versions(self, path: str) -> int:
        return self._cpio.versions(path)

    @overrides(MountSource)
    def open(self, fileInfo: FileInfo, buffering: int = -1) -> IO[bytes]:
        return self._cpio.open(fileInfo, buffering=buffering)

    @overrides(MountSource)
    def read(self, fileInfo: FileInfo, size: int, offset: int) -> bytes:
        return self._cpio.read(fileInfo, size, offset)

    def __enter__(self):
        return self

    @overrides(MountSource)
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._cpio.close()
        if not self.isFileObject and getattr(self, "fileObject", None) is not None:
            self.fileObject.close()
