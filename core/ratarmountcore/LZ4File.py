"""Seekable LZ4 frame reader with block index for random access.

Uses the official LZ4 Frame format:
https://github.com/lz4/lz4/blob/dev/doc/lz4_Frame_format.md

The frame-level block dictionary size and independence flag come from the
frame descriptor. Independent blocks can be decompressed from their start;
dependent blocks require decoding from an earlier independent boundary
(or the frame start). Per-block content is handled by lz4.block.
"""

from __future__ import annotations

import io
import logging
import struct
from dataclasses import dataclass
from typing import IO, BinaryIO, Optional, Union

from .utils import RatarmountError, overrides

logger = logging.getLogger(__name__)

try:
    import lz4.block as _lz4_block
except ImportError:  # pragma: no cover
    _lz4_block = None  # type: ignore

LZ4_FRAME_MAGIC = 0x184D2204
LZ4_SKIPPABLE_MASK = 0x184D2A50
LZ4_SKIPPABLE_NIBBLE = 0xFFFF_FFF0

# FLG bits
_FLG_DICT_ID = 0x01
_FLG_CONTENT_CHECKSUM = 0x04
_FLG_CONTENT_SIZE = 0x08
_FLG_BLOCK_CHECKSUM = 0x10
_FLG_BLOCK_INDEP = 0x20


class LZ4Error(RatarmountError):
    """Raised when an LZ4 frame cannot be parsed or decompressed."""


@dataclass
class LZ4BlockInfo:
    """One data block within an LZ4 frame."""

    # Absolute offset of the 4-byte block-size field in the file.
    size_field_offset: int
    # Absolute offset of compressed (or raw) block payload.
    data_offset: int
    compressed_size: int
    uncompressed_offset: int
    uncompressed_size: int
    is_uncompressed: bool


@dataclass
class LZ4FrameInfo:
    start_offset: int  # magic offset
    data_start: int  # first block size field
    block_independence: bool
    block_checksum: bool
    content_checksum: bool
    content_size: Optional[int]
    blocks: list[LZ4BlockInfo]
    # Offset just past the end mark (and optional content checksum).
    end_offset: int
    total_uncompressed: int


def _require_lz4() -> None:
    if _lz4_block is None:
        raise LZ4Error("The 'lz4' package is required for LZ4 support. Install with: pip install lz4")


def _read_exact(fileobj: IO[bytes], n: int) -> bytes:
    data = fileobj.read(n)
    if len(data) != n:
        raise LZ4Error(f"Unexpected EOF (wanted {n} bytes, got {len(data)})")
    return data


def skip_skippable_frames(fileobj: IO[bytes]) -> None:
    """Advance fileobj past any leading LZ4 skippable frames."""
    while True:
        pos = fileobj.tell()
        header = fileobj.read(4)
        if len(header) < 4:
            fileobj.seek(pos)
            return
        (magic,) = struct.unpack("<I", header)
        if magic & LZ4_SKIPPABLE_NIBBLE != LZ4_SKIPPABLE_MASK:
            fileobj.seek(pos)
            return
        (frame_size,) = struct.unpack("<I", _read_exact(fileobj, 4))
        fileobj.seek(fileobj.tell() + frame_size)


def parse_lz4_frame(fileobj: IO[bytes]) -> LZ4FrameInfo:
    """Parse one LZ4 frame at the current position and index its blocks."""
    _require_lz4()
    start = fileobj.tell()
    (magic,) = struct.unpack("<I", _read_exact(fileobj, 4))
    if magic != LZ4_FRAME_MAGIC:
        raise LZ4Error(f"Invalid LZ4 frame magic at {start}: 0x{magic:08x}")

    flg = _read_exact(fileobj, 1)[0]
    bd = _read_exact(fileobj, 1)[0]
    version = (flg >> 6) & 0x03
    if version != 1:
        raise LZ4Error(f"Unsupported LZ4 frame version: {version}")

    block_independence = bool(flg & _FLG_BLOCK_INDEP)
    block_checksum = bool(flg & _FLG_BLOCK_CHECKSUM)
    content_size_flag = bool(flg & _FLG_CONTENT_SIZE)
    content_checksum = bool(flg & _FLG_CONTENT_CHECKSUM)
    dict_id_flag = bool(flg & _FLG_DICT_ID)

    content_size = None
    if content_size_flag:
        (content_size,) = struct.unpack("<Q", _read_exact(fileobj, 8))
    if dict_id_flag:
        _read_exact(fileobj, 4)  # dictionary ID ignored for now

    # Header checksum (xxHash of descriptor) — 1 byte; we skip validation.
    _read_exact(fileobj, 1)

    # Block maximum size from BD bits 4-5 (only informational for reading).
    _max_block_size_code = (bd >> 4) & 0x07

    blocks: list[LZ4BlockInfo] = []
    uncompressed_offset = 0
    data_start = fileobj.tell()

    while True:
        size_field_offset = fileobj.tell()
        (block_header,) = struct.unpack("<I", _read_exact(fileobj, 4))
        if block_header == 0:
            # EndMark
            break
        is_uncompressed = bool(block_header & 0x8000_0000)
        compressed_size = block_header & 0x7FFF_FFFF
        data_offset = fileobj.tell()
        block_data = _read_exact(fileobj, compressed_size)
        if block_checksum:
            _read_exact(fileobj, 4)

        if is_uncompressed:
            uncompressed_size = compressed_size
        elif block_independence:
            # Independent blocks: decompress once while indexing to learn size.
            # Cost is comparable to building a gzip seek index.
            plain = _decompress_lz4_block(block_data)
            uncompressed_size = len(plain)
        else:
            # Dependent blocks need a running dictionary; sizes filled in a second pass.
            uncompressed_size = -1

        blocks.append(
            LZ4BlockInfo(
                size_field_offset=size_field_offset,
                data_offset=data_offset,
                compressed_size=compressed_size,
                uncompressed_offset=uncompressed_offset if uncompressed_size >= 0 else -1,
                uncompressed_size=uncompressed_size,
                is_uncompressed=is_uncompressed,
            )
        )
        if uncompressed_size >= 0:
            uncompressed_offset += uncompressed_size

    if content_checksum:
        _read_exact(fileobj, 4)

    end_offset = fileobj.tell()

    # Dependent frames: one full decompress to assign uncompressed offsets/sizes.
    if not block_independence and blocks:
        import lz4.frame

        fileobj.seek(start)
        compressed = _read_exact(fileobj, end_offset - start)
        plain = lz4.frame.decompress(compressed)
        # Without per-block sizes from the stream, treat the frame as one logical range
        # for seeking (still O(frame) for random access, but correct).
        if len(blocks) == 1:
            blocks[0] = LZ4BlockInfo(
                size_field_offset=blocks[0].size_field_offset,
                data_offset=blocks[0].data_offset,
                compressed_size=blocks[0].compressed_size,
                uncompressed_offset=0,
                uncompressed_size=len(plain),
                is_uncompressed=blocks[0].is_uncompressed,
            )
        else:
            # Approximate equal split only if unknown — better: re-decode block-by-block
            # with lz4.frame decompressor feeding chunks is complex; store whole frame size.
            total = len(plain)
            # Keep block compressed locations; set cumulative sizes by re-decompressing
            # each block with lz4.block is invalid without dict. Collapse to one synthetic block.
            blocks = [
                LZ4BlockInfo(
                    size_field_offset=blocks[0].size_field_offset,
                    data_offset=blocks[0].data_offset,
                    compressed_size=end_offset - data_start,
                    uncompressed_offset=0,
                    uncompressed_size=total,
                    is_uncompressed=False,
                )
            ]
        uncompressed_offset = len(plain)
        fileobj.seek(end_offset)

    if content_size is not None and content_size != uncompressed_offset:
        logger.debug(
            "LZ4 content size header %s differs from sum of blocks %s", content_size, uncompressed_offset
        )

    return LZ4FrameInfo(
        start_offset=start,
        data_start=data_start,
        block_independence=block_independence,
        block_checksum=block_checksum,
        content_checksum=content_checksum,
        content_size=content_size,
        blocks=blocks,
        end_offset=end_offset,
        total_uncompressed=uncompressed_offset,
    )


def _decompress_lz4_block(block_data: bytes) -> bytes:
    assert _lz4_block is not None
    # lz4.block requires an upper bound; try growing sizes.
    for bound in (256 * 1024, 1024 * 1024, 8 * 1024 * 1024, 64 * 1024 * 1024):
        try:
            return _lz4_block.decompress(block_data, uncompressed_size=bound)
        except _lz4_block.LZ4BlockError:
            continue
        except Exception:
            continue
    # Last resort without bound (older API)
    return _lz4_block.decompress(block_data)


def index_lz4_file(fileobj: IO[bytes]) -> list[LZ4FrameInfo]:
    """Index all LZ4 frames in a file (skippable frames are skipped)."""
    fileobj.seek(0)
    frames: list[LZ4FrameInfo] = []
    while True:
        skip_skippable_frames(fileobj)
        pos = fileobj.tell()
        header = fileobj.read(4)
        if len(header) < 4:
            break
        fileobj.seek(pos)
        (magic,) = struct.unpack("<I", header)
        if magic != LZ4_FRAME_MAGIC:
            break
        frames.append(parse_lz4_frame(fileobj))
    if not frames:
        raise LZ4Error("No LZ4 frames found")
    return frames


class IndexedLZ4File(io.RawIOBase):
    """Seekable read-only view of an LZ4 frame stream."""

    def __init__(self, fileobj: Union[str, IO[bytes]], **_kwargs):
        super().__init__()
        _require_lz4()
        self._close_file = False
        if isinstance(fileobj, str):
            self._file: IO[bytes] = open(fileobj, "rb")
            self._close_file = True
        else:
            self._file = fileobj
        self._frames = index_lz4_file(self._file)
        self._size = sum(f.total_uncompressed for f in self._frames)
        # Cumulative uncompressed sizes per frame for multi-frame files.
        self._frame_starts: list[int] = []
        total = 0
        for frame in self._frames:
            self._frame_starts.append(total)
            total += frame.total_uncompressed
        self._pos = 0
        # Cache of last decompressed block: (frame_idx, block_idx, data)
        self._block_cache: Optional[tuple[int, int, bytes]] = None

    @property
    def size(self) -> int:
        return self._size

    def fileno(self) -> int:
        return self._file.fileno()

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
        return self._pos

    @overrides(io.RawIOBase)
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new < 0:
            raise ValueError("negative seek")
        self._pos = min(new, self._size)
        return self._pos

    def _find_frame_block(self, pos: int) -> tuple[int, int, int]:
        """Return (frame_index, block_index, offset_within_block)."""
        if pos >= self._size:
            return len(self._frames) - 1, max(0, len(self._frames[-1].blocks) - 1), 0
        for fi, frame in enumerate(self._frames):
            start = self._frame_starts[fi]
            if pos < start + frame.total_uncompressed:
                local = pos - start
                for bi, block in enumerate(frame.blocks):
                    if local < block.uncompressed_offset + block.uncompressed_size:
                        return fi, bi, local - block.uncompressed_offset
                # Past last block due to rounding — clamp
                if frame.blocks:
                    last = frame.blocks[-1]
                    return fi, len(frame.blocks) - 1, last.uncompressed_size
        raise LZ4Error(f"Position {pos} out of range (size {self._size})")

    def _decompress_block(self, frame_idx: int, block_idx: int) -> bytes:
        if self._block_cache and self._block_cache[0] == frame_idx and self._block_cache[1] == block_idx:
            return self._block_cache[2]

        frame = self._frames[frame_idx]
        block = frame.blocks[block_idx]
        self._file.seek(block.data_offset)
        payload = _read_exact(self._file, block.compressed_size)
        if block.is_uncompressed:
            data = payload
        else:
            assert _lz4_block is not None
            data = _lz4_block.decompress(payload, uncompressed_size=block.uncompressed_size)
            if len(data) != block.uncompressed_size:
                data = data[: block.uncompressed_size]
        self._block_cache = (frame_idx, block_idx, data)
        return data

    def _decode_from_block(self, frame_idx: int, start_block: int, skip: int, length: int) -> bytes:
        """Decompress from start_block, skip `skip` uncompressed bytes, take `length`."""
        frame = self._frames[frame_idx]
        if frame.block_independence:
            # Can start at start_block directly.
            out = bytearray()
            remaining_skip = skip
            remaining = length
            bi = start_block
            while remaining > 0 and bi < len(frame.blocks):
                data = self._decompress_block(frame_idx, bi)
                if remaining_skip >= len(data):
                    remaining_skip -= len(data)
                    bi += 1
                    continue
                chunk = data[remaining_skip : remaining_skip + remaining]
                remaining_skip = 0
                out.extend(chunk)
                remaining -= len(chunk)
                bi += 1
            return bytes(out)

        # Dependent blocks: decode from block 0 (or last independent — always 0 if none).
        # For simplicity and correctness, start from the beginning of the frame when
        # block independence is false. Still use the block index to stop early.
        out = bytearray()
        remaining_skip = frame.blocks[start_block].uncompressed_offset + skip
        remaining = length
        # Absolute uncompressed offset where we want to start within the frame:
        start_u = frame.blocks[start_block].uncompressed_offset + skip
        end_u = start_u + length
        for bi, block in enumerate(frame.blocks):
            if block.uncompressed_offset + block.uncompressed_size <= start_u:
                # Still need to decompress for dictionary if dependent — must decode.
                if not frame.block_independence:
                    self._decompress_block(frame_idx, bi)
                continue
            if block.uncompressed_offset >= end_u:
                break
            data = self._decompress_block(frame_idx, bi)
            # Dependent lz4 blocks: python-lz4 block.decompress doesn't accept dict.
            # For dependent streams we need lz4.frame streaming instead.
            # Fallback: if not independent, re-read whole frame via lz4.frame for the range.
            _ = data
            break
        else:
            return b""

        # Dependent path: use frame decompressor from frame start.
        return self._decode_dependent_range(frame_idx, start_u, length)

    def _decode_dependent_range(self, frame_idx: int, start_u: int, length: int) -> bytes:
        import lz4.frame

        frame = self._frames[frame_idx]
        self._file.seek(frame.start_offset)
        # Read whole frame compressed region through end_offset
        compressed = _read_exact(self._file, frame.end_offset - frame.start_offset)
        plain = lz4.frame.decompress(compressed)
        return plain[start_u : start_u + length]

    @overrides(io.RawIOBase)
    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        if size is None or size < 0:
            size = self._size - self._pos
        size = min(size, self._size - self._pos)
        if size == 0:
            return b""

        fi, bi, within = self._find_frame_block(self._pos)
        frame = self._frames[fi]

        if frame.block_independence:
            data = self._decode_from_block(fi, bi, within, size)
        else:
            local_pos = self._pos - self._frame_starts[fi]
            data = self._decode_dependent_range(fi, local_pos, size)

        self._pos += len(data)
        return data

    @overrides(io.RawIOBase)
    def readinto(self, b) -> int:  # type: ignore[override]
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    @overrides(io.RawIOBase)
    def close(self) -> None:
        if self._close_file and self._file is not None:
            self._file.close()
        super().close()


def open_lz4_file(fileobj: Union[str, IO[bytes]], **kwargs) -> IndexedLZ4File:
    """Open an LZ4 file as a seekable file object (API for COMPRESSION_BACKENDS)."""
    return IndexedLZ4File(fileobj, **kwargs)
