"""
Tests for /experimental/qwen3vlembeddings video embedding endpoint.

Requires pre-converted Qwen3-VL-Embedding-2B GGUF files in models/.
See the build instructions in the implementation plan.
"""

import base64
import io
import math

import pytest

from utils import ServerPreset, ServerProcess

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.qwen3vl_embedding()


def _make_test_image(width: int = 28, height: int = 28, color: tuple = (128, 64, 32)) -> str:
    """Generate a small solid-color BMP image and return as base64 string."""
    # minimal BMP: 14-byte file header + 40-byte DIB header + pixel data
    row_size = (width * 3 + 3) & ~3  # rows padded to 4-byte boundary
    pixel_data_size = row_size * height
    file_size = 54 + pixel_data_size

    bmp = bytearray()
    # BMP file header (14 bytes)
    bmp += b"BM"
    bmp += file_size.to_bytes(4, "little")
    bmp += (0).to_bytes(2, "little")  # reserved
    bmp += (0).to_bytes(2, "little")  # reserved
    bmp += (54).to_bytes(4, "little")  # pixel data offset
    # DIB header (40 bytes, BITMAPINFOHEADER)
    bmp += (40).to_bytes(4, "little")  # header size
    bmp += width.to_bytes(4, "little", signed=True)
    bmp += height.to_bytes(4, "little", signed=True)
    bmp += (1).to_bytes(2, "little")   # color planes
    bmp += (24).to_bytes(2, "little")  # bits per pixel
    bmp += (0).to_bytes(4, "little")   # compression (none)
    bmp += pixel_data_size.to_bytes(4, "little")
    bmp += (2835).to_bytes(4, "little")  # horizontal resolution
    bmp += (2835).to_bytes(4, "little")  # vertical resolution
    bmp += (0).to_bytes(4, "little")   # colors in palette
    bmp += (0).to_bytes(4, "little")   # important colors

    # pixel data (BMP is bottom-up, BGR order)
    r, g, b = color
    for _ in range(height):
        row = bytearray()
        for _ in range(width):
            row += bytes([b, g, r])
        # pad row to 4-byte boundary
        while len(row) % 4 != 0:
            row += b"\x00"
        bmp += row

    return base64.b64encode(bytes(bmp)).decode("utf-8")


def _base64_test_image() -> str:
    return _make_test_image()


# --- v1 backward compatibility ---


def test_qwen3vl_text_embedding():
    """qwen3vlembeddings delegates to v1 for text-only input."""
    server.start()
    res = server.make_request("POST", "/experimental/qwen3vlembeddings", data={
        "input": "hello world",
    })
    assert res.status_code == 200
    assert "data" in res.body
    assert len(res.body["data"]) == 1
    emb = res.body["data"][0]["embedding"]
    assert len(emb) > 0


# --- Video embedding ---


def test_video_segments_basic():
    """15 frames at 1fps -> 2 segments with default 10s duration, 5s stride."""
    server.start()
    img = _base64_test_image()
    frames = [{"timestamp": float(i), "image": img} for i in range(15)]
    res = server.make_request("POST", "/experimental/qwen3vlembeddings", data={
        "video_frames": frames,
        "segment_duration": 10.0,
        "segment_stride": 5.0,
    })
    assert res.status_code == 200
    body = res.body
    assert body["total_segments"] == 2
    assert body["total_frames"] == 15
    assert len(body["data"]) == 2
    # each segment should have an embedding vector
    assert len(body["data"][0]["embedding"]) > 0
    # final embedding is the mean-pooled, L2-normalized result
    assert len(body["final_embedding"]) > 0
    # verify segment metadata
    assert body["data"][0]["segment"]["start_time"] == 0.0
    assert body["data"][0]["segment"]["end_time"] == 10.0


def test_video_single_segment():
    """3s video with default 10s window -> 1 segment."""
    server.start()
    img = _base64_test_image()
    frames = [{"timestamp": float(i), "image": img} for i in range(3)]
    res = server.make_request("POST", "/experimental/qwen3vlembeddings", data={
        "video_frames": frames,
    })
    assert res.status_code == 200
    assert res.body["total_segments"] == 1


def test_video_embedding_normalized():
    """Video final embedding is L2-normalized."""
    server.start()
    img = _base64_test_image()
    frames = [{"timestamp": float(i), "image": img} for i in range(5)]
    res = server.make_request("POST", "/experimental/qwen3vlembeddings", data={
        "video_frames": frames,
    })
    assert res.status_code == 200
    vec = res.body["final_embedding"]
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4


def test_video_with_instruction():
    """Video embedding with instruction prompt."""
    server.start()
    img = _base64_test_image()
    frames = [{"timestamp": float(i), "image": img} for i in range(4)]
    res = server.make_request("POST", "/experimental/qwen3vlembeddings", data={
        "video_frames": frames,
        "instruction": "Describe the visual content of this video.",
    })
    assert res.status_code == 200
    assert len(res.body["final_embedding"]) > 0


# --- Error cases ---


def test_video_too_many_frames():
    """More than 30 frames returns 400."""
    server.start()
    img = _base64_test_image()
    frames = [{"timestamp": float(i), "image": img} for i in range(35)]
    res = server.make_request("POST", "/experimental/qwen3vlembeddings", data={
        "video_frames": frames,
    })
    assert res.status_code == 400


def test_video_stride_gt_duration():
    """stride > duration returns 400."""
    server.start()
    img = _base64_test_image()
    frames = [{"timestamp": float(i), "image": img} for i in range(5)]
    res = server.make_request("POST", "/experimental/qwen3vlembeddings", data={
        "video_frames": frames,
        "segment_duration": 5.0,
        "segment_stride": 10.0,
    })
    assert res.status_code == 400


def test_video_missing_timestamp():
    """Frames without timestamp return 400."""
    server.start()
    img = _base64_test_image()
    res = server.make_request("POST", "/experimental/qwen3vlembeddings", data={
        "video_frames": [{"image": img}],
    })
    assert res.status_code == 400


def test_video_empty_frames():
    """Empty video_frames array returns 400."""
    server.start()
    res = server.make_request("POST", "/experimental/qwen3vlembeddings", data={
        "video_frames": [],
    })
    assert res.status_code == 400
