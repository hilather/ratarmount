"""XAR archive MountSource with random access via TOC heap offsets."""

from __future__ import annotations

import struct
import stat
import zlib
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Optional, Union

from ratarmountcore.mountsource.formats.stenciled import StenciledArchiveMountSource, make_file_row
from ratarmountcore.SQLiteIndex import SQLiteIndex
from ratarmountcore.utils import RatarmountError


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _walk_xar_files(element: ET.Element, prefix: str, heap_offset: int, rows: list[tuple]) -> None:
    for child in element:
        if _local(child.tag) != "file":
            continue
        name_el = next((c for c in child if _local(c.tag) == "name"), None)
        name = name_el.text if name_el is not None and name_el.text else "unnamed"
        full = f"{prefix}/{name}" if prefix else name
        full = full.lstrip("/")

        type_el = next((c for c in child if _local(c.tag) == "type"), None)
        file_type = (type_el.text or "file") if type_el is not None else "file"

        if file_type == "directory":
            path, base = SQLiteIndex.normpath(full).rsplit("/", 1)
            rows.append(
                make_file_row(
                    path=path,
                    name=base,
                    header_offset=0,
                    data_offset=0,
                    size=0,
                    mtime=0,
                    mode=0o755 | stat.S_IFDIR,
                )
            )
            _walk_xar_files(child, full, heap_offset, rows)
            continue

        data_el = next((c for c in child if _local(c.tag) == "data"), None)
        if data_el is None:
            continue

        length = 0
        offset = 0
        size = 0
        encoding = "application/octet-stream"
        for field in data_el:
            tag = _local(field.tag)
            text = (field.text or "0").strip()
            if tag == "length":
                length = int(text)
            elif tag == "offset":
                offset = int(text)
            elif tag == "size":
                size = int(text)
            elif tag == "encoding":
                encoding = field.attrib.get("style") or field.attrib.get("name") or text

        # Only stored (uncompressed) members get true stencil random access for now.
        # Compressed members still get listed; open will fail with a clear error if encoding unsupported.
        data_offset = heap_offset + offset
        path, base = SQLiteIndex.normpath(full).rsplit("/", 1)
        # Encode encoding style into linkname for open() to detect compressed members.
        # Use size as uncompressed size for listing; stored open uses length when encoding is identity.
        is_stored = encoding in (
            "application/octet-stream",
            "application/x-gzip",  # sometimes misused; treat carefully below
            "",
        )
        # XAR "encoding style" for none is typically application/octet-stream
        style = encoding
        rows.append(
            make_file_row(
                path=path,
                name=base,
                header_offset=data_offset,
                data_offset=data_offset,
                size=size if size else length,
                mtime=0,
                mode=0o644 | stat.S_IFREG,
                linkname=f"xar-enc:{style}|packed:{length}",
            )
        )


def parse_xar_archive(fileobj: IO[bytes]) -> list[tuple]:
    header = fileobj.read(28)
    if len(header) < 28 or header[:4] != b"xar!":
        raise RatarmountError("Not a XAR archive")

    header_size, _version, toc_comp_len, toc_uncomp_len, _checksum_algo = struct.unpack(">HHQQI", header[4:28])
    # Some headers are 28 bytes; header_size field may be 28.
    if header_size < 28:
        header_size = 28

    fileobj.seek(header_size)
    toc_compressed = fileobj.read(toc_comp_len)
    if len(toc_compressed) < toc_comp_len:
        raise RatarmountError("Truncated XAR TOC")
    try:
        toc_xml = zlib.decompress(toc_compressed)
    except zlib.error as exc:
        raise RatarmountError("Failed to decompress XAR TOC") from exc
    if toc_uncomp_len and len(toc_xml) < toc_uncomp_len:
        # Allow exact match variation
        pass

    heap_offset = header_size + toc_comp_len
    try:
        root = ET.fromstring(toc_xml)
    except ET.ParseError as exc:
        raise RatarmountError("Invalid XAR TOC XML") from exc

    # TOC root is <xar><toc>...</toc>
    toc = root
    if _local(root.tag) == "xar":
        toc = next((c for c in root if _local(c.tag) == "toc"), root)

    rows: list[tuple] = []
    _walk_xar_files(toc, "", heap_offset, rows)
    if not rows:
        raise RatarmountError("XAR archive contains no files")
    return rows


class XARMountSource(StenciledArchiveMountSource):
    def __init__(self, fileOrPath: Union[str, IO[bytes], Path], **options) -> None:
        def build_rows(fileobj: IO[bytes]) -> Iterable[tuple]:
            fileobj.seek(0)
            return parse_xar_archive(fileobj)

        super().__init__(fileOrPath, backendName="XARMountSource", build_rows=build_rows, **options)

    def open(self, fileInfo, buffering: int = -1):
        import io

        from ratarmountcore.mountsource import FileInfo as _FileInfo  # local for typing clarity
        from ratarmountcore.SQLiteIndex import SQLiteIndex as _SQLiteIndex

        link = fileInfo.linkname or ""
        if not link.startswith("xar-enc:"):
            return super().open(fileInfo, buffering=buffering)

        style = link[len("xar-enc:") :].split("|", 1)[0]
        if style in ("application/octet-stream", ""):
            # Stored: use packed length from metadata when present.
            if "|packed:" in link:
                try:
                    packed_size = int(link.rsplit("|packed:", 1)[1])
                    offset = _SQLiteIndex.get_index_userdata(fileInfo.userdata).offset
                    return self._open_stencil(offset, packed_size, buffering)
                except ValueError:
                    pass
            return super().open(fileInfo, buffering=buffering)

        if style in ("application/x-gzip", "application/gzip", "application/x-bzip2"):
            packed_size = fileInfo.size
            if "|packed:" in link:
                try:
                    packed_size = int(link.rsplit("|packed:", 1)[1])
                except ValueError:
                    pass
            offset = _SQLiteIndex.get_index_userdata(fileInfo.userdata).offset
            with self.fileObjectLock:
                self.fileObject.seek(offset)
                packed = self.fileObject.read(packed_size)
            if style == "application/x-bzip2":
                import bz2

                data = bz2.decompress(packed)
            else:
                try:
                    data = zlib.decompress(packed)
                except zlib.error:
                    data = zlib.decompress(packed, wbits=16 + zlib.MAX_WBITS)
            return io.BytesIO(data)

        raise RatarmountError(f"XAR member encoding not supported for random access: {style}")
