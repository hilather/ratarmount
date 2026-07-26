# pylint: disable=wrong-import-order,wrong-import-position,protected-access

import os
import sys

import pytest
from helpers import find_test_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("lz4.frame")
from ratarmountcore.compressions import detect_compression, open_compressed_file  # noqa: E402
from ratarmountcore.formats import FID  # noqa: E402
from ratarmountcore.LZ4File import IndexedLZ4File  # noqa: E402
from ratarmountcore.mountsource.compositing.automount import AutoMountLayer  # noqa: E402
from ratarmountcore.mountsource.factory import open_mount_source  # noqa: E402


def _require(name: str) -> str:
    path = find_test_file(name)
    if not os.path.isfile(path):
        pytest.skip(f"missing {name}")
    return path


class TestIndexedLZ4File:
    def test_simple_lz4(self):
        path = _require("simple.lz4")
        with IndexedLZ4File(path) as file:
            assert file.read() == b"foo fighter\n"
            file.seek(4)
            assert file.read() == b"fighter\n"
            file.seek(0, os.SEEK_END)
            assert file.tell() == 12

    def test_skippable_frame_prefix(self):
        path = _require("nested-tar.skippable-frame.lz4")
        with IndexedLZ4File(path) as file:
            assert file.size > 0
            data = file.read()
            assert len(data) == file.size
            file.seek(100)
            assert file.read(50) == data[100:150]

    def test_multiblock_independent(self):
        path = _require("multiblock-independent.lz4")
        with IndexedLZ4File(path) as file:
            assert file._frames[0].block_independence
            assert len(file._frames[0].blocks) > 1
            full = file.read()
            mid = len(full) // 2
            file.seek(mid)
            assert file.read(64) == full[mid : mid + 64]
            file.seek(max(0, len(full) - 32))
            assert file.read() == full[-32:]

    def test_multiblock_dependent(self):
        path = _require("multiblock-dependent.lz4")
        with IndexedLZ4File(path) as file:
            full = file.read()
            file.seek(1000)
            assert file.read(40) == full[1000:1040]

    def test_open_compressed_file_backend(self):
        path = _require("simple.lz4")
        with open(path, "rb") as raw:
            assert detect_compression(raw) == FID.LZ4
            raw.seek(0)
            decomp, _raw, comp = open_compressed_file(raw)
            assert comp == FID.LZ4
            assert type(decomp).__name__ == "IndexedLZ4File"
            assert decomp.read() == b"foo fighter\n"

    def test_tar_lz4_recursive_mount(self):
        path = _require("nested-tar.skippable-frame.lz4")
        options = {"recursive": True, "indexFilePath": ":memory:", "clearIndexCache": True}
        # File is lz4-compressed tar
        with open(path, "rb") as raw:
            _decomp, _, comp = open_compressed_file(raw)
            assert comp == FID.LZ4
        base = open_mount_source(path, **options)
        try:
            auto = AutoMountLayer(base, **options)
            # nested-tar structure: should expose foo/ or similar
            root = auto.list("/")
            assert root is not None
            # Seekable decompression allows random access into tar without full extract first
            names = list(root.keys()) if isinstance(root, dict) else list(root)
            assert names
        finally:
            base.close()
