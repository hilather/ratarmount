"""Shared base for archive formats that open members via StenciledFile (offset + size)."""

from __future__ import annotations

import contextlib
import os
import stat
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Optional, Union, cast

from ratarmountcore.mountsource import FileInfo, MountSource
from ratarmountcore.mountsource.SQLiteIndexMountSource import SQLiteIndexMountSource
from ratarmountcore.SQLiteIndex import SQLiteIndex
from ratarmountcore.StenciledFile import RawStenciledFile, StenciledFile
from ratarmountcore.utils import RatarmountError, overrides


class StenciledArchiveMountSource(SQLiteIndexMountSource):
    """MountSource base for formats that store members contiguously at known offsets.

    Subclasses parse the archive once, call ``_finalize_index`` with SQLite rows whose
    ``offset`` column is the absolute data offset and ``size`` is the payload size.
    """

    def __init__(
        self,
        fileOrPath: Union[str, IO[bytes], Path],
        *,
        backendName: str,
        build_rows: Callable[[IO[bytes]], Iterable[tuple]],
        **options,
    ) -> None:
        if isinstance(fileOrPath, Path):
            fileOrPath = str(fileOrPath)
        self.isFileObject = not isinstance(fileOrPath, str)
        self.fileObject: IO[bytes] = open(fileOrPath, "rb") if isinstance(fileOrPath, str) else fileOrPath

        indexOptions = {
            "archiveFilePath": fileOrPath if isinstance(fileOrPath, str) else None,
            "backendName": backendName,
            **{k: v for k, v in options.items() if k != "indexFilePath"},
        }
        if "indexFilePath" in options:
            indexOptions["indexFilePath"] = options["indexFilePath"]

        super().__init__(**indexOptions)

        self.blockSize = 512
        with contextlib.suppress(Exception):
            self.blockSize = os.fstat(self.fileObject.fileno()).st_blksize

        self.fileObjectLock = threading.Lock()
        self._finalize_index(lambda: self.index.set_file_infos(list(build_rows(self.fileObject))))

    def _open_stencil(self, offset: int, size: int, buffering: int) -> IO[bytes]:
        if buffering == 0:
            return cast(IO[bytes], RawStenciledFile([(self.fileObject, offset, size)], self.fileObjectLock))
        return cast(
            IO[bytes],
            StenciledFile(
                [(self.fileObject, offset, size)],
                self.fileObjectLock,
                bufferSize=self.blockSize if buffering == -1 else buffering,
            ),
        )

    @overrides(MountSource)
    def open(self, fileInfo: FileInfo, buffering: int = -1) -> IO[bytes]:
        if stat.S_ISDIR(fileInfo.mode):
            raise RatarmountError("Cannot open directory as file")
        if stat.S_ISLNK(fileInfo.mode):
            raise RatarmountError("Cannot read contents of symbolic link!")
        if fileInfo.size == 0:
            import io

            return cast(IO[bytes], io.BytesIO(b""))
        offset = SQLiteIndex.get_index_userdata(fileInfo.userdata).offset
        return self._open_stencil(offset, fileInfo.size, buffering)

    @overrides(SQLiteIndexMountSource)
    def close(self) -> None:
        super().close()
        if not self.isFileObject and getattr(self, "fileObject", None) is not None:
            self.fileObject.close()
            self.fileObject = None  # type: ignore[assignment]


def make_file_row(
    *,
    path: str,
    name: str,
    header_offset: int,
    data_offset: int,
    size: int,
    mtime: float,
    mode: int,
    linkname: str = "",
    uid: int = 0,
    gid: int = 0,
) -> tuple:
    """Build a standard SQLiteIndex files-table row tuple."""
    # fmt: off
    return (
        path,           # 0  path
        name,           # 1  name
        header_offset,  # 2  offsetheader
        data_offset,    # 3  offset (data)
        size,           # 4  size
        mtime,          # 5  mtime
        mode,           # 6  mode
        0,              # 7  type
        linkname,       # 8  linkname
        uid,            # 9  uid
        gid,            # 10 gid
        False,          # 11 isTar
        False,          # 12 isSparse
        False,          # 13 isGenerated
        0,              # 14 recursion depth
    )
    # fmt: on
