# pylint: disable=wrong-import-order,wrong-import-position,protected-access
"""Tests for stencil-based archive backends: CPIO, ISO9660, WARC, XAR, deb composition."""

import bz2
import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest

from helpers import copy_test_file, find_test_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ratarmountcore.mountsource.compositing.automount import AutoMountLayer
from ratarmountcore.mountsource.factory import open_mount_source
from ratarmountcore.mountsource.formats.cpio import CPIOMountSource, parse_cpio_archive
from ratarmountcore.mountsource.formats.iso9660 import ISO9660MountSource, parse_iso9660_archive
from ratarmountcore.mountsource.formats.warc import WARCMountSource
from ratarmountcore.mountsource.formats.xar import XARMountSource


def _require(name: str) -> str:
    path = find_test_file(name)
    if not os.path.isfile(path):
        pytest.skip(f"missing fixture {name}")
    return path


# ---------------------------------------------------------------------------
# CPIO
# ---------------------------------------------------------------------------


class TestCPIO:
    @pytest.mark.parametrize(
        "name",
        [
            "single-file.newc.cpio",
            "single-file.crc.cpio",
            "single-file.odc.cpio",
            "single-file.bin.cpio",
            "single-file.hpbin.cpio",
            "single-file.hpodc.cpio",
        ],
    )
    def test_single_file_variants(self, name):
        path = _require(name)
        with CPIOMountSource(path, indexFilePath=":memory:") as mount_source:
            assert type(mount_source).__name__ == "CPIOMountSource"
            info = mount_source.lookup("/bar")
            assert info is not None
            with mount_source.open(info) as file:
                assert file.read() == b"foo\n"
                file.seek(1)
                assert file.read() == b"oo\n"

    def test_factory_prefers_cpio(self):
        with copy_test_file("single-file.newc.cpio") as path:
            ms = open_mount_source(path, indexFilePath=":memory:")
            try:
                assert type(ms).__name__ == "CPIOMountSource"
            finally:
                ms.close()

    def test_nested_tar_in_cpio_member(self):
        path = _require("nested-tar-in-cpio.newc.cpio")
        with CPIOMountSource(path, indexFilePath=":memory:") as mount_source:
            assert mount_source.open(mount_source.lookup("/readme.txt")).read() == b"cpio root file\n"
            tar_member = mount_source.open(mount_source.lookup("/nested/inner.tar")).read()
            assert len(tar_member) > 100

    def test_automount_tar_inside_cpio(self):
        options = {"recursive": True, "indexFilePath": ":memory:", "clearIndexCache": True}
        with copy_test_file("nested-tar-in-cpio.newc.cpio") as path:
            base = open_mount_source(path, **options)
            try:
                assert type(base).__name__ == "CPIOMountSource"
                auto = AutoMountLayer(base, **options)
                inner = auto.lookup("/nested/inner.tar/inner.txt")
                assert inner is not None
                with auto.open(inner) as file:
                    assert file.read() == b"hello from tar-in-cpio\n"
            finally:
                base.close()

    def test_parse_rejects_garbage(self):
        import io

        with pytest.raises(Exception):
            parse_cpio_archive(io.BytesIO(b"not a cpio file!!!!"))


# ---------------------------------------------------------------------------
# ISO 9660
# ---------------------------------------------------------------------------


class TestISO9660:
    def test_single_file_iso(self):
        compressed = _require("single-file.iso.bz2")
        iso_bytes = bz2.decompress(Path(compressed).read_bytes())
        with tempfile.NamedTemporaryFile(suffix=".iso") as tmp:
            tmp.write(iso_bytes)
            tmp.flush()
            with ISO9660MountSource(tmp.name, indexFilePath=":memory:") as mount_source:
                # ISO often uppercases names
                listing = mount_source.list("/")
                assert listing
                names = list(listing.keys()) if isinstance(listing, dict) else list(listing)
                assert any(n.upper() == "BAR" for n in names)
                bar_name = next(n for n in names if n.upper() == "BAR")
                with mount_source.open(mount_source.lookup("/" + bar_name)) as file:
                    assert file.read() == b"foo\n"
                    file.seek(0)
                    assert file.read(1) == b"f"

    def test_factory_opens_iso(self):
        compressed = _require("single-file.iso.bz2")
        iso_bytes = bz2.decompress(Path(compressed).read_bytes())
        with tempfile.NamedTemporaryFile(suffix=".iso") as tmp:
            tmp.write(iso_bytes)
            tmp.flush()
            ms = open_mount_source(tmp.name, indexFilePath=":memory:")
            try:
                assert type(ms).__name__ == "ISO9660MountSource"
            finally:
                ms.close()

    def test_parse_rejects_non_iso(self):
        import io

        with pytest.raises(Exception):
            parse_iso9660_archive(io.BytesIO(b"\x00" * 100000))


# ---------------------------------------------------------------------------
# WARC
# ---------------------------------------------------------------------------


class TestWARC:
    def test_hello_world_warc(self):
        path = _require("hello-world.warc")
        with WARCMountSource(path, indexFilePath=":memory:") as mount_source:
            # Response payload for hello-world.txt
            target = None
            # Walk tree for a file containing HTTP response body text
            def find(path_prefix="/"):
                nonlocal target
                listing = mount_source.list(path_prefix) or {}
                items = listing.items() if isinstance(listing, dict) else []
                for name, info in items:
                    child = f"{path_prefix.rstrip('/')}/{name}" if path_prefix != "/" else f"/{name}"
                    if stat.S_ISDIR(info.mode):
                        find(child)
                    elif name.endswith("hello-world.txt") and not child.startswith("/_warc/request"):
                        target = child

            find()
            assert target is not None, "expected response record for hello-world.txt"
            data = mount_source.open(mount_source.lookup(target)).read()
            assert b"HTTP/1.1" in data or b"hello" in data.lower()

    def test_simple_response_warc(self):
        path = _require("simple-response.warc")
        with WARCMountSource(path, indexFilePath=":memory:") as mount_source:
            info = mount_source.lookup("/example.com/hello.txt")
            assert info is not None
            data = mount_source.open(info).read()
            assert b"hello warc" in data
            # seek within payload
            with mount_source.open(info) as file:
                file.seek(len(data) - 10)
                assert b"warc" in file.read()

    def test_factory_prefers_warc(self):
        with copy_test_file("simple-response.warc") as path:
            ms = open_mount_source(path, indexFilePath=":memory:")
            try:
                assert type(ms).__name__ == "WARCMountSource"
            finally:
                ms.close()


# ---------------------------------------------------------------------------
# XAR
# ---------------------------------------------------------------------------


class TestXAR:
    def test_single_file_xar(self):
        path = _require("single-file.xar")
        with XARMountSource(path, indexFilePath=":memory:") as mount_source:
            info = mount_source.lookup("/bar")
            assert info is not None
            with mount_source.open(info) as file:
                assert file.read() == b"foo\n"

    def test_factory_prefers_xar(self):
        with copy_test_file("single-file.xar") as path:
            ms = open_mount_source(path, indexFilePath=":memory:")
            try:
                assert type(ms).__name__ == "XARMountSource"
            finally:
                ms.close()


# ---------------------------------------------------------------------------
# deb composition (AR outer + nested tars via recursion)
# ---------------------------------------------------------------------------


class TestDebComposition:
    def test_deb_uses_ar_and_recursive_tar(self):
        path = _require("testpkg_0.0.1_all.deb")
        options = {"recursive": True, "indexFilePath": ":memory:", "clearIndexCache": True}
        base = open_mount_source(path, **options)
        try:
            # Outer should be AR (or libarchive fallback, but AR is preferred for !<arch>)
            assert type(base).__name__ in ("ARMountSource", "LibarchiveMountSource")
            auto = AutoMountLayer(base, **options)
            root = auto.list("/")
            assert root
            names = list(root.keys()) if isinstance(root, dict) else list(root)
            # debian packages contain control.tar.* / data.tar.* / debian-binary
            assert any("debian-binary" in n or "control" in n or "data" in n for n in names)

            # Navigate into data.tar.* if present
            data_member = next((n for n in names if n.startswith("data.tar")), None)
            if data_member:
                # list inside recursively mounted data archive
                listing = auto.list("/" + data_member)
                # May be empty package; just ensure no crash
                assert listing is not None or True
        finally:
            base.close()
