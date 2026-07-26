"""WARC MountSource with random access via record payload offsets."""

from __future__ import annotations

import re
import stat
from typing import IO, TYPE_CHECKING, Union
from urllib.parse import urlparse

from ratarmountcore.mountsource.formats.stenciled import StenciledArchiveMountSource, make_file_row
from ratarmountcore.SQLiteIndex import SQLiteIndex
from ratarmountcore.utils import RatarmountError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_CONTENT_LENGTH_RE = re.compile(rb"(?im)^Content-Length:\s*(\d+)\s*$")
_WARC_TYPE_RE = re.compile(rb"(?im)^WARC-Type:\s*(\S+)\s*$")
_WARC_URI_RE = re.compile(rb"(?im)^WARC-Target-URI:\s*(\S+)\s*$")
_WARC_RECORD_ID_RE = re.compile(rb"(?im)^WARC-Record-ID:\s*<?([^>\r\n]+)>?\s*$")


def _uri_to_path(uri: str) -> str:
    uri = uri.strip().replace("\\", "/")
    if "://" in uri:
        parsed = urlparse(uri)
        host = parsed.netloc or "unknown-host"
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        uri = f"{host}{path}"
    return uri.lstrip("/") or "index"


def _sanitize_name(name: str, used: dict[str, int]) -> str:
    name = _uri_to_path(name)
    if not name or name.endswith("/"):
        name = (name or "record") + "index.html"
    # Avoid collisions for multiple records with same URI.
    key = name
    if key in used:
        used[key] += 1
        stem, dot, ext = name.rpartition(".")
        name = f"{stem}-{used[key]}.{ext}" if dot and stem else f"{name}-{used[key]}"
    else:
        used[key] = 0
    return name


def parse_warc_archive(fileobj: IO[bytes]) -> list[tuple]:
    """Parse WARC/1.x records into SQLite rows for payload bodies."""
    data = fileobj.read()
    if not data.startswith(b"WARC/"):
        raise RatarmountError("Not a WARC file (missing WARC/ version line)")

    rows: list[tuple] = []
    used_names: dict[str, int] = {}
    pos = 0
    record_index = 0

    while pos < len(data):
        # Skip blank lines between records.
        while pos < len(data) and data[pos : pos + 1] in (b"\r", b"\n"):
            if data[pos : pos + 2] == b"\r\n":
                pos += 2
            else:
                pos += 1

        if pos >= len(data):
            break
        if not data[pos:].startswith(b"WARC/"):
            # Trailing garbage or padding.
            break

        header_start = pos
        # Find end of WARC headers (blank line).
        sep = data.find(b"\r\n\r\n", pos)
        if sep < 0:
            sep = data.find(b"\n\n", pos)
            if sep < 0:
                raise RatarmountError(f"WARC record at {pos} missing header terminator")
            header_blob = data[pos:sep]
            payload_offset = sep + 2
        else:
            header_blob = data[pos:sep]
            payload_offset = sep + 4

        length_match = _CONTENT_LENGTH_RE.search(header_blob)
        if not length_match:
            raise RatarmountError(f"WARC record at {header_start} missing Content-Length")
        content_length = int(length_match.group(1))

        type_match = _WARC_TYPE_RE.search(header_blob)
        warc_type = type_match.group(1).decode("ascii", errors="replace").lower() if type_match else "unknown"

        uri_match = _WARC_URI_RE.search(header_blob)
        record_id_match = _WARC_RECORD_ID_RE.search(header_blob)

        # Expose response/resource payloads as files; still index warcinfo/request as metadata files.
        if uri_match:
            display = _uri_to_path(uri_match.group(1).decode("utf-8", errors="replace"))
        elif record_id_match:
            display = record_id_match.group(1).decode("utf-8", errors="replace").replace(":", "_")
        else:
            display = f"record-{record_index}"

        if warc_type in ("warcinfo", "request", "metadata", "revisit", "conversion"):
            display = f"_warc/{warc_type}/{display}"
        display = _sanitize_name(display, used_names)

        if payload_offset + content_length > len(data):
            raise RatarmountError(f"WARC record at {header_start} truncated payload")

        path, name = SQLiteIndex.normpath(display).rsplit("/", 1)
        rows.append(
            make_file_row(
                path=path,
                name=name,
                header_offset=header_start,
                data_offset=payload_offset,
                size=content_length,
                mtime=0,
                mode=0o644 | stat.S_IFREG,
            )
        )

        # Payload followed by CRLF CRLF typically.
        pos = payload_offset + content_length
        record_index += 1

    if not rows:
        raise RatarmountError("WARC file contains no records")
    return rows


class WARCMountSource(StenciledArchiveMountSource):
    def __init__(self, fileOrPath: str | IO[bytes] | Path, **options) -> None:
        def build_rows(fileobj: IO[bytes]) -> Iterable[tuple]:
            fileobj.seek(0)
            return parse_warc_archive(fileobj)

        super().__init__(fileOrPath, backendName="WARCMountSource", build_rows=build_rows, **options)
