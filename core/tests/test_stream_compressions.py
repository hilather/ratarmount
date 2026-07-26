# pylint: disable=wrong-import-order,wrong-import-position

import os
import sys

import pytest
from helpers import find_test_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ratarmountcore.compressions import detect_compression, open_compressed_file
from ratarmountcore.formats import FID


def _require(name: str) -> str:
    path = find_test_file(name)
    if not os.path.isfile(path):
        pytest.skip(f"missing {name}")
    return path


def _assert_seekable_plain(path: str, expected: bytes, backend_name: str):
    with open(path, "rb") as raw:
        decomp, _raw, _comp = open_compressed_file(raw)
        assert type(decomp).__name__ == backend_name
        assert decomp.read() == expected
        decomp.seek(4)
        assert decomp.read() == expected[4:]
        decomp.seek(0)
        assert decomp.read(3) == expected[:3]


class TestLZO:
    def test_simple_lzo(self):
        path = _require("simple.lzo")
        from ratarmountcore.LZOFile import parse_lzop_file

        with open(path, "rb") as f:
            info = parse_lzop_file(f)
        assert info.total_uncompressed == 12
        assert len(info.blocks) >= 1
        _assert_seekable_plain(path, b"foo fighter\n", "IndexedLZOFile")

    def test_detect_lzop(self):
        path = _require("simple.lzo")
        with open(path, "rb") as raw:
            assert detect_compression(raw) == FID.LZOP

    def test_multiblock_lzo_if_present(self):
        path = find_test_file("multiblock.lzo")
        if not os.path.isfile(path):
            pytest.skip("multiblock.lzo not generated (need lzop CLI)")
        from ratarmountcore.LZOFile import IndexedLZOFile

        with IndexedLZOFile(path) as file:
            assert file.size > 100_000
            assert len(file._info.blocks) > 1  # pylint: disable=protected-access
            full_head = file.read(100)
            file.seek(file.size // 2)
            mid = file.read(50)
            file.seek(0)
            assert file.read(100) == full_head
            assert len(mid) == 50


class TestCompressZ:
    def test_simple_z(self):
        path = _require("simple.Z")
        pytest.importorskip("unlzw3")
        _assert_seekable_plain(path, b"foo fighter\n", "IndexedCompressZFile")

    def test_detect_z(self):
        path = _require("simple.Z")
        with open(path, "rb") as raw:
            assert detect_compression(raw) == FID.Z


class TestUU:
    def test_classic_uu(self):
        path = _require("simple.uu")
        expected = b"hello world from uu\n" * 5
        _assert_seekable_plain(path, expected, "IndexedUUFile")

    def test_base64_uu(self):
        path = _require("simple-base64.uu")
        expected = b"hello world from uu\n" * 5
        _assert_seekable_plain(path, expected, "IndexedUUFile")

    def test_detect_uu(self):
        path = _require("simple.uu")
        with open(path, "rb") as raw:
            assert detect_compression(raw) == FID.UU


class TestLzip:
    def test_simple_lzip(self):
        path = _require("simple.lzip")
        from ratarmountcore.LzipFile import IndexedLzipFile

        with IndexedLzipFile(path) as file:
            assert file.read() == b"foo fighter\n"
            file.seek(4)
            assert file.read() == b"fighter\n"
        _assert_seekable_plain(path, b"foo fighter\n", "IndexedLzipFile")

    def test_detect_lzip(self):
        path = _require("simple.lzip")
        with open(path, "rb") as raw:
            assert detect_compression(raw) == FID.LZIP


class TestLZMAAlone:
    def test_simple_lzma(self):
        path = _require("simple.lzma")
        _assert_seekable_plain(path, b"foo fighter\n", "IndexedLZMAFile")

    def test_detect_lzma(self):
        path = _require("simple.lzma")
        with open(path, "rb") as raw:
            assert detect_compression(raw) == FID.LZMA
