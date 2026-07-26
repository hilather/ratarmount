import contextlib
import datetime
import logging
import stat
import struct
import sys
import threading
import zipfile
from pathlib import Path
from typing import IO, Optional, Union, cast

from ratarmountcore.mountsource import FileInfo, MountSource
from ratarmountcore.mountsource.SQLiteIndexMountSource import SQLiteIndexMountSource
from ratarmountcore.SQLiteIndex import SQLiteIndex
from ratarmountcore.StenciledFile import RawStenciledFile, StenciledFile
from ratarmountcore.utils import overrides

try:
    # Importing this patches the zipfile module as a "side" effect!
    import fast_zip_decryption  # pylint: disable=unused-import  # noqa: F401
except (ImportError, Exception):
    with contextlib.suppress(ImportError, Exception):
        import fastzipfile  # pylint: disable=unused-import  # noqa: F401


logger = logging.getLogger(__name__)


class ZipMountSource(SQLiteIndexMountSource):
    def __init__(self, fileOrPath: Union[str, IO[bytes], Path], **options) -> None:
        if 'zipfile' not in sys.modules:
            raise RuntimeError("Did not find the zipfile module. Please use Python 3.7+.")

        if isinstance(fileOrPath, Path):
            fileOrPath = str(fileOrPath)
        self.fileObject = zipfile.ZipFile(fileOrPath, 'r')
        # Underlying archive stream for STORE stencils (path or ZipFile.fp).
        self._archiveFile: Optional[IO[bytes]] = None
        self._archiveLock = threading.Lock()
        if isinstance(fileOrPath, str):
            self._archivePath: Optional[str] = fileOrPath
        else:
            self._archivePath = None
            self._archiveFile = fileOrPath if hasattr(fileOrPath, "seek") else getattr(self.fileObject, "fp", None)

        ZipMountSource._find_password(self.fileObject, options.get("passwords", []))
        self.files = {info.header_offset: info for info in self.fileObject.infolist()}
        # header_offset → absolute data offset for uncompressed (STORE) members.
        self._storeDataOffsets: dict[int, int] = {}
        for info in self.fileObject.infolist():
            if info.compress_type == zipfile.ZIP_STORED and not info.is_dir() and info.file_size > 0:
                with contextlib.suppress(Exception):
                    self._storeDataOffsets[info.header_offset] = self._local_data_offset(info)

        indexOptions = {
            'archiveFilePath': fileOrPath if isinstance(fileOrPath, str) else None,
            'backendName': 'ZipMountSource',
        }
        super().__init__(**(options | indexOptions))
        self._finalize_index(
            lambda: self.index.set_file_infos([self._convert_to_row(info) for info in self.fileObject.infolist()])
        )

    def _local_data_offset(self, info: "zipfile.ZipInfo") -> int:
        """Absolute offset of file data after the local ZIP header."""
        # Local header: 30 bytes fixed + filename + extra
        # We trust ZipInfo.header_offset; filename length may differ from central dir
        # (general purpose bit 11 etc.) — read the local header.
        zf = self.fileObject
        fp = zf.fp
        assert fp is not None
        with self._archiveLock:
            pos = fp.tell()
            try:
                fp.seek(info.header_offset)
                local = fp.read(30)
                if len(local) < 30 or local[:4] != b"PK\x03\x04":
                    # Fallback: central-dir name/extra lengths
                    encoding = getattr(zf, "metadata_encoding", None) or "utf-8"
                    return info.header_offset + 30 + len(info.filename.encode(encoding)) + len(info.extra or b"")
                _sig, _ver, _flags, _method, _time, _date, _crc, _csize, _usize, nlen, elen = struct.unpack(
                    "<IHHHHHIIIHH", local
                )
                return info.header_offset + 30 + nlen + elen
            finally:
                fp.seek(pos)

    def _convert_to_row(self, info: "zipfile.ZipInfo") -> tuple:
        mtime = datetime.datetime(*info.date_time, tzinfo=datetime.timezone.utc).timestamp() if info.date_time else 0

        # According to section 4.5.7 in the .ZIP file format specification, links are supported:
        # https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
        # The Python zipfile module has no API for links: https://bugs.python.org/issue45286
        # However, the file mode exposes whether it's a link and the file mode is shown by ZipInfo.__repr__.
        # For that, it uses the OS-dependent external_attr member. See also the ZIP specification on that:
        # > 4.4.15 external file attributes: (4 bytes)
        # >   The mapping of the external attributes is host-system dependent (see 'version made by').
        # >   For MS-DOS, the low order byte is the MS-DOS directory attribute byte.
        # >   If input came from standard input, this field is set to zero.

        # file_redir is (type, flags, target) or None. Only tested for type == RAR5_XREDIR_UNIX_SYMLINK.
        linkname = ""
        mode = (info.external_attr >> 16) & 0o777
        if mode == 0:
            mode = 0o770 if info.is_dir() else 0o660
        if stat.S_ISLNK(info.external_attr >> 16):
            linkname = self.fileObject.read(info).decode()
            mode = mode | stat.S_IFLNK
        else:
            mode = mode | (stat.S_IFDIR if info.is_dir() else stat.S_IFREG)

        path, name = SQLiteIndex.normpath(self.transform(info.filename)).rsplit("/", 1)

        # Currently, this is unused. The index only is used for getting metadata. (The data offset
        # is already determined and written out in order to possibly speed up reading of encrypted
        # files by implementing the decryption ourselves.)
        # The data offset is deprecated again! Collecting it can add a huge overhead for large zip files
        # because we have to seek to every position and read a few bytes from it. Furthermore, it is useless
        # by itself anyway. We don't even store yet how the data is compressed or encrypted, so we would
        # have to read the local header again anyway!
        dataOffset = 0

        # fmt: off
        fileInfo : tuple = (
            path              ,  # 0  : path
            name              ,  # 1  : file name
            info.header_offset,  # 2  : header offset
            dataOffset        ,  # 3  : data offset
            info.file_size    ,  # 4  : file size
            mtime             ,  # 5  : modification time
            mode              ,  # 6  : file mode / permissions
            0                 ,  # 7  : TAR file type. Currently unused. Overlaps with mode
            linkname          ,  # 8  : linkname
            0                 ,  # 9  : user ID
            0                 ,  # 10 : group ID
            False             ,  # 11 : is TAR (unused?)
            False             ,  # 12 : is sparse
            False             ,  # 13 : is generated (parent folder)
            0                 ,  # 14 : recursion depth
        )
        # fmt: on

        return fileInfo

    @staticmethod
    def _find_password(fileobj: "zipfile.ZipFile", passwords: list[bytes]) -> Optional[bytes]:
        # If headers are encrypted, then infolist will simply return an empty list!
        files = fileobj.infolist()
        if not files:
            for password in passwords:
                fileobj.setpassword(password)
                files = fileobj.infolist()
                if files:
                    return password

        # If headers are not encrypted, then try out passwords by trying to open the first file.
        files = [file for file in files if not file.is_dir() and file.file_size > 0]
        if not files:
            return None

        for password_or_none in [None, *passwords]:
            if password_or_none:
                fileobj.setpassword(password_or_none)
            try:
                with fileobj.open(files[0]) as file:
                    file.read(1)
                return password_or_none
            except Exception:
                pass

        raise RuntimeError("Could not find a matching password!")

    @overrides(SQLiteIndexMountSource)
    def close(self) -> None:
        super().close()
        if fileObject := getattr(self, 'fileObject', None):
            fileObject.close()

    def _open_store_stencil(self, info: "zipfile.ZipInfo", buffering: int) -> Optional[IO[bytes]]:
        data_offset = self._storeDataOffsets.get(info.header_offset)
        if data_offset is None or info.file_size <= 0:
            return None
        # Prefer opening a dedicated handle for path-based archives so ZipFile.fp is undisturbed.
        if self._archivePath is not None:
            archive = open(self._archivePath, "rb")
            lock = threading.Lock()

            def _close_archive():
                archive.close()

            if buffering == 0:
                f: Union[RawStenciledFile, StenciledFile] = RawStenciledFile(
                    [(archive, data_offset, info.file_size)], lock
                )
            else:
                f = StenciledFile(
                    [(archive, data_offset, info.file_size)],
                    lock,
                    bufferSize=65536 if buffering < 0 else buffering,
                )
            # Close underlying archive when stencil closes
            original_close = f.close

            def close_and_release():
                original_close()
                _close_archive()

            f.close = close_and_release  # type: ignore[method-assign]
            return cast(IO[bytes], f)

        fp = self._archiveFile or getattr(self.fileObject, "fp", None)
        if fp is None:
            return None
        if buffering == 0:
            return cast(IO[bytes], RawStenciledFile([(fp, data_offset, info.file_size)], self._archiveLock))
        return cast(
            IO[bytes],
            StenciledFile(
                [(fp, data_offset, info.file_size)],
                self._archiveLock,
                bufferSize=65536 if buffering < 0 else buffering,
            ),
        )

    @overrides(MountSource)
    def open(self, fileInfo: FileInfo, buffering=-1) -> IO[bytes]:
        info = self.files[SQLiteIndex.get_index_userdata(fileInfo.userdata).offsetheader]
        assert isinstance(info, zipfile.ZipInfo)
        # True random access for STORE (uncompressed) members via file stencil.
        if info.compress_type == zipfile.ZIP_STORED and not info.is_dir():
            stenciled = self._open_store_stencil(info, buffering)
            if stenciled is not None:
                return stenciled
        # Deflate / encrypted / etc.: zipfile handles per-member inflate.
        # https://github.com/python/cpython/blob/a87c46eab3c306b1c5b8a072b7b30ac2c50651c0/Lib/zipfile/__init__.py#L1569
        return self.fileObject.open(info, 'r')  # https://github.com/pauldmccarthy/indexed_gzip/issues/85
