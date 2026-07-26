"""Microsoft CAB MountSource with random access for store and MSZIP folders.

Cabinet format (MS-CAB): CFHEADER → CFFOLDER[] → CFFILE[] → CFDATA[].
Files reference a folder and an uncompressed offset within that folder's stream.

- typeCompress 0 (none): true stencil open of file spans across CFDATA blocks
- typeCompress 1 (MSZIP): CFDATA blocks decompressed (CK + deflate); file slices
  taken from the reconstructed folder stream (cached per folder)
- typeCompress 2/3 (Quantum/LZX): not supported — raise so factory can fall back
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import stat
import struct
import threading
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Union, cast

from ratarmountcore.mountsource import FileInfo, MountSource
from ratarmountcore.mountsource.SQLiteIndexMountSource import SQLiteIndexMountSource
from ratarmountcore.mountsource.formats.stenciled import make_file_row
from ratarmountcore.SQLiteIndex import SQLiteIndex
from ratarmountcore.StenciledFile import RawStenciledFile, StenciledFile
from ratarmountcore.utils import RatarmountError, overrides

logger = logging.getLogger(__name__)

TCOMP_MASK_TYPE = 0x000F
TCOMP_TYPE_NONE = 0x0000
TCOMP_TYPE_MSZIP = 0x0001
TCOMP_TYPE_QUANTUM = 0x0002
TCOMP_TYPE_LZX = 0x0003

# CFFILE attributes
_A_RDONLY = 0x01
_A_HIDDEN = 0x02
_A_SYSTEM = 0x04
_A_ATTRIB = 0x08
_A_DIRECTORY = 0x10
_A_ARCHIVE = 0x20
_A_NAME_IS_UTF = 0x80

_MSZIP_WINDOW = 32768


class CABError(RatarmountError):
    pass


@dataclass
class CFDataBlock:
    offset: int  # absolute offset of compressed payload (after CFDATA header + reserve)
    compressed_size: int
    uncompressed_size: int
    uncompressed_offset: int  # within folder stream


@dataclass
class CFFolder:
    index: int
    data_offset: int  # first CFDATA
    num_data: int
    type_compress: int
    blocks: list[CFDataBlock] = field(default_factory=list)
    total_uncompressed: int = 0


@dataclass
class CFFile:
    name: str
    size: int
    folder_index: int
    folder_offset: int  # uncompressed offset in folder
    attributes: int
    header_offset: int
    mtime: float = 0.0


@dataclass
class CABArchive:
    folders: list[CFFolder]
    files: list[CFFile]


def _read_exact(fileobj: IO[bytes], n: int) -> bytes:
    data = fileobj.read(n)
    if len(data) != n:
        raise CABError("Truncated CAB data")
    return data


def _read_cstring(fileobj: IO[bytes]) -> bytes:
    parts: list[bytes] = []
    while True:
        b = fileobj.read(1)
        if not b:
            raise CABError("Truncated CAB string")
        if b == b"\x00":
            break
        parts.append(b)
    return b"".join(parts)


def _cab_dos_datetime_to_mtime(date: int, time: int) -> float:
    """Convert FAT/DOS date+time to Unix timestamp (best-effort, local)."""
    if date == 0 and time == 0:
        return 0.0
    day = date & 0x1F
    month = (date >> 5) & 0x0F
    year = ((date >> 9) & 0x7F) + 1980
    second = (time & 0x1F) * 2
    minute = (time >> 5) & 0x3F
    hour = (time >> 11) & 0x1F
    try:
        import calendar
        import datetime

        dt = datetime.datetime(year, month, day, hour, minute, second)
        return float(calendar.timegm(dt.timetuple()))
    except (ValueError, OverflowError):
        return 0.0


def parse_cab_archive(fileobj: IO[bytes]) -> CABArchive:
    """Parse CAB headers and index CFDATA block locations (does not load payloads)."""
    start = fileobj.tell()
    header = _read_exact(fileobj, 36)
    if header[:4] != b"MSCF":
        raise CABError("Not a Microsoft CAB file")

    _sig, _res1, _cb_cabinet, _res2, coff_files, _res3 = struct.unpack_from("<4sIIIII", header, 0)
    ver_min, ver_maj, c_folders, c_files, flags, _set_id, _i_cabinet = struct.unpack_from("<BBHHHHH", header, 24)
    if ver_maj != 1:
        raise CABError(f"Unsupported CAB major version: {ver_maj}.{ver_min}")

    cb_cf_folder = 0
    cb_cf_data = 0
    if flags & 0x0004:  # cfhdrRESERVE_PRESENT
        res = _read_exact(fileobj, 4)
        cb_cf_header, cb_cf_folder, cb_cf_data = struct.unpack("<HBB", res)
        if cb_cf_header:
            _read_exact(fileobj, cb_cf_header)
    if flags & 0x0001:  # prev cabinet
        _read_cstring(fileobj)
        _read_cstring(fileobj)
    if flags & 0x0002:  # next cabinet
        _read_cstring(fileobj)
        _read_cstring(fileobj)

    folders: list[CFFolder] = []
    for i in range(c_folders):
        raw = _read_exact(fileobj, 8)
        coff_cab_start, c_cf_data, type_compress = struct.unpack("<IHH", raw)
        if cb_cf_folder:
            _read_exact(fileobj, cb_cf_folder)
        folders.append(
            CFFolder(
                index=i,
                data_offset=start + coff_cab_start,
                num_data=c_cf_data,
                type_compress=type_compress & TCOMP_MASK_TYPE,
            )
        )

    files: list[CFFile] = []
    fileobj.seek(start + coff_files)
    for _ in range(c_files):
        header_offset = fileobj.tell() - start
        raw = _read_exact(fileobj, 16)
        # CFFILE: u4 cbFile, u4 uoffFolderStart, u2 iFolder, u2 date, u2 time, u2 attribs
        cb_file, uoff_folder_start, i_folder, date, time, attribs = struct.unpack("<IIHHHH", raw)
        name_raw = _read_cstring(fileobj)
        if attribs & _A_NAME_IS_UTF:
            name = name_raw.decode("utf-8", errors="replace")
        else:
            name = name_raw.decode("latin-1", errors="replace")
        # iFolder special values: 0xFFFD continued from prev, 0xFFFE continued to next, 0xFFFF both
        if i_folder >= 0xFFFD:
            raise CABError(f"Split CAB file spans not supported: {name!r} iFolder=0x{i_folder:04x}")
        if i_folder >= len(folders):
            raise CABError(f"Invalid folder index {i_folder} for file {name!r}")
        files.append(
            CFFile(
                name=name.replace("\\", "/"),
                size=cb_file,
                folder_index=i_folder,
                folder_offset=uoff_folder_start,
                attributes=attribs,
                header_offset=header_offset,
                mtime=_cab_dos_datetime_to_mtime(date, time),
            )
        )

    # Index CFDATA blocks per folder (payload offsets only)
    for folder in folders:
        fileobj.seek(folder.data_offset)
        u_off = 0
        for _ in range(folder.num_data):
            raw = _read_exact(fileobj, 8)
            _csum, cb_data, cb_uncomp = struct.unpack("<IHH", raw)
            if cb_cf_data:
                _read_exact(fileobj, cb_cf_data)
            payload_offset = fileobj.tell()
            folder.blocks.append(
                CFDataBlock(
                    offset=payload_offset,
                    compressed_size=cb_data,
                    uncompressed_size=cb_uncomp,
                    uncompressed_offset=u_off,
                )
            )
            fileobj.seek(payload_offset + cb_data)
            u_off += cb_uncomp
        folder.total_uncompressed = u_off

    return CABArchive(folders=folders, files=files)


def _mszip_decompress_block(block: bytes, uncompressed_size: int, history: bytes) -> bytes:
    """Decompress one MSZIP CFDATA payload, using prior folder window as zlib dictionary when available."""
    if len(block) < 2 or block[:2] != b"CK":
        raise CABError("Invalid MSZIP block (missing CK signature)")
    payload = block[2:]

    def _raw_inflate(data: bytes, max_out: int) -> bytes:
        dobj = zlib.decompressobj(wbits=-15)
        out = dobj.decompress(data, max_out)
        out += dobj.flush()
        return out[:max_out] if max_out else out

    # Prefer dictionary when the platform supports it (MSZIP 32 KiB window across blocks).
    if history and hasattr(zlib.decompressobj(wbits=-15), "set_dictionary"):
        try:
            dobj = zlib.decompressobj(wbits=-15)
            dobj.set_dictionary(history[-_MSZIP_WINDOW:])
            out = dobj.decompress(payload, uncompressed_size)
            out += dobj.flush()
            return out[:uncompressed_size] if uncompressed_size else out
        except (zlib.error, AttributeError):
            pass
    try:
        return _raw_inflate(payload, uncompressed_size)
    except zlib.error as exc:
        raise CABError(f"MSZIP decompress failed: {exc}") from exc


class CABMountSource(SQLiteIndexMountSource):
    def __init__(self, fileOrPath: Union[str, IO[bytes], Path], **options) -> None:
        if isinstance(fileOrPath, Path):
            fileOrPath = str(fileOrPath)
        self.isFileObject = not isinstance(fileOrPath, str)
        self.fileObject: IO[bytes] = open(fileOrPath, "rb") if isinstance(fileOrPath, str) else fileOrPath
        self.fileObject.seek(0)
        self._cab = parse_cab_archive(self.fileObject)
        # Reject entirely if any folder uses unsupported compression — fall back via factory.
        for folder in self._cab.folders:
            if folder.type_compress not in (TCOMP_TYPE_NONE, TCOMP_TYPE_MSZIP):
                raise CABError(
                    f"CAB compression type {folder.type_compress} not supported by custom backend "
                    f"(only none/MSZIP); use libarchive"
                )

        self._folder_plain_cache: dict[int, bytes] = {}
        self.blockSize = 512
        with contextlib.suppress(Exception):
            self.blockSize = os.fstat(self.fileObject.fileno()).st_blksize  # type: ignore[union-attr]
        self.fileObjectLock = threading.Lock()

        indexOptions = {
            "archiveFilePath": fileOrPath if isinstance(fileOrPath, str) else None,
            "backendName": "CABMountSource",
            **{k: v for k, v in options.items() if k != "indexFilePath"},
        }
        if "indexFilePath" in options:
            indexOptions["indexFilePath"] = options["indexFilePath"]
        super().__init__(**indexOptions)
        self._finalize_index(lambda: self.index.set_file_infos(list(self._build_rows())))

    def _build_rows(self) -> Iterable[tuple]:
        rows = []
        transform = self.transform
        for f in self._cab.files:
            full = transform(f.name)
            path, name = SQLiteIndex.normpath(full).rsplit("/", 1)
            is_dir = bool(f.attributes & _A_DIRECTORY)
            mode = (0o755 | stat.S_IFDIR) if is_dir else (0o644 | stat.S_IFREG)
            rows.append(
                make_file_row(
                    path=path,
                    name=name,
                    header_offset=f.folder_index,
                    data_offset=f.folder_offset,
                    size=0 if is_dir else f.size,
                    mtime=f.mtime,
                    mode=mode,
                    linkname=f"cab-folder:{f.folder_index}",
                )
            )
        return rows

    def _folder_bytes(self, folder_index: int) -> bytes:
        if folder_index in self._folder_plain_cache:
            return self._folder_plain_cache[folder_index]
        folder = self._cab.folders[folder_index]
        if folder.type_compress == TCOMP_TYPE_NONE:
            parts = []
            with self.fileObjectLock:
                for block in folder.blocks:
                    self.fileObject.seek(block.offset)
                    parts.append(self.fileObject.read(block.compressed_size))
            plain = b"".join(parts)
        elif folder.type_compress == TCOMP_TYPE_MSZIP:
            parts: list[bytes] = []
            history = b""
            with self.fileObjectLock:
                for block in folder.blocks:
                    self.fileObject.seek(block.offset)
                    raw = self.fileObject.read(block.compressed_size)
                    chunk = _mszip_decompress_block(raw, block.uncompressed_size, history)
                    parts.append(chunk)
                    history = (history + chunk)[-_MSZIP_WINDOW:]
            plain = b"".join(parts)
        else:
            raise CABError(f"Unsupported CAB compression {folder.type_compress}")
        self._folder_plain_cache[folder_index] = plain
        return plain

    def _open_store_file(self, folder_index: int, folder_offset: int, size: int, buffering: int) -> IO[bytes]:
        """Map a file in an uncompressed folder to stencils over CFDATA payloads."""
        folder = self._cab.folders[folder_index]
        stencils: list[tuple] = []
        remaining = size
        pos = folder_offset
        for block in folder.blocks:
            block_start = block.uncompressed_offset
            block_end = block_start + block.uncompressed_size
            if remaining <= 0 or pos >= block_end:
                if pos >= block_end:
                    continue
            if pos < block_start:
                continue
            local = pos - block_start
            take = min(remaining, block.uncompressed_size - local)
            if take <= 0:
                continue
            stencils.append((self.fileObject, block.offset + local, take))
            pos += take
            remaining -= take
        if remaining != 0 or not stencils:
            data = self._folder_bytes(folder_index)[folder_offset : folder_offset + size]
            return cast(IO[bytes], io.BytesIO(data))
        if buffering == 0:
            return cast(IO[bytes], RawStenciledFile(stencils, self.fileObjectLock))
        return cast(
            IO[bytes],
            StenciledFile(stencils, self.fileObjectLock, bufferSize=self.blockSize if buffering == -1 else buffering),
        )

    @overrides(MountSource)
    def open(self, fileInfo: FileInfo, buffering: int = -1) -> IO[bytes]:
        if stat.S_ISDIR(fileInfo.mode):
            raise RatarmountError("Cannot open directory as file")
        if fileInfo.size == 0:
            return cast(IO[bytes], io.BytesIO(b""))
        user = SQLiteIndex.get_index_userdata(fileInfo.userdata)
        folder_index = int(user.offsetheader)
        folder_offset = int(user.offset)
        folder = self._cab.folders[folder_index]
        if folder.type_compress == TCOMP_TYPE_NONE:
            return self._open_store_file(folder_index, folder_offset, fileInfo.size, buffering)
        # MSZIP: slice from cached folder stream
        plain = self._folder_bytes(folder_index)
        return cast(IO[bytes], io.BytesIO(plain[folder_offset : folder_offset + fileInfo.size]))

    @overrides(SQLiteIndexMountSource)
    def close(self) -> None:
        super().close()
        self._folder_plain_cache.clear()
        if not self.isFileObject and getattr(self, "fileObject", None) is not None:
            self.fileObject.close()
            self.fileObject = None  # type: ignore[assignment]
