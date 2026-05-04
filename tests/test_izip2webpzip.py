"""Tests for izip2webpzip."""
import io
import zipfile

import pytest
from PIL import Image

import izip2webpzip


def _make_zip(images: dict[str, Image.Image], format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, img in images.items():
            img_buf = io.BytesIO()
            img.save(img_buf, format=format)
            zf.writestr(name, img_buf.getvalue())
    return buf.getvalue()


def _entries(zip_bytes: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def test_basic_conversion() -> None:
    src = _make_zip({"01.png": Image.new("RGB", (50, 50), "red")})
    out = izip2webpzip.convert(src)

    entries = _entries(out)
    assert "01.webp" in entries
    assert entries["01.webp"][:4] == b"RIFF"
    assert b"WEBP" in entries["01.webp"][:12]


def test_path_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src_path = tmp_path / "in.zip"
    src_path.write_bytes(_make_zip({"a.png": Image.new("RGB", (50, 50), "blue")}))
    out_path = tmp_path / "out.zip"

    out = izip2webpzip.convert(str(src_path), str(out_path))

    assert out_path.exists()
    assert out == out_path.read_bytes()
    assert "a.webp" in _entries(out)


def test_multiple_images_natsorted() -> None:
    src = _make_zip({
        "10.png": Image.new("RGB", (10, 10), "red"),
        "2.png": Image.new("RGB", (10, 10), "green"),
        "1.png": Image.new("RGB", (10, 10), "blue"),
    })
    out = izip2webpzip.convert(src)

    with zipfile.ZipFile(io.BytesIO(out)) as zf:
        names = zf.namelist()
    assert names == ["1.webp", "2.webp", "10.webp"]


def test_lossless_mode_produces_webp() -> None:
    src = _make_zip({"01.png": Image.new("RGB", (50, 50), "red")})
    out = izip2webpzip.convert(src, lossless=True)
    assert _entries(out)["01.webp"][:4] == b"RIFF"


def test_quality_affects_size() -> None:
    img = Image.new("RGB", (200, 200))
    for x in range(200):
        for y in range(200):
            img.putpixel((x, y), (x, y, (x + y) % 256))
    src = _make_zip({"01.png": img})

    low = izip2webpzip.convert(src, quality=10)
    high = izip2webpzip.convert(src, quality=95)

    assert len(_entries(low)["01.webp"]) < len(_entries(high)["01.webp"])


def test_directory_entries_skipped() -> None:
    buf = io.BytesIO()
    img_buf = io.BytesIO()
    Image.new("RGB", (50, 50), "red").save(img_buf, format="PNG")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("subdir/", b"")
        zf.writestr("subdir/01.png", img_buf.getvalue())
    src = buf.getvalue()

    out = izip2webpzip.convert(src)
    entries = _entries(out)
    assert "subdir/01.webp" in entries
    assert "subdir/" not in entries  # directory entry skipped


def test_on_error_keep_preserves_original() -> None:
    buf = io.BytesIO()
    img_buf = io.BytesIO()
    Image.new("RGB", (50, 50), "red").save(img_buf, format="PNG")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("01.png", img_buf.getvalue())
        zf.writestr("readme.txt", b"not an image")
    src = buf.getvalue()

    out = izip2webpzip.convert(src, on_error="keep")
    entries = _entries(out)

    assert "01.webp" in entries
    assert "readme.txt" in entries
    assert entries["readme.txt"] == b"not an image"


def test_on_error_skip_drops_failing() -> None:
    buf = io.BytesIO()
    img_buf = io.BytesIO()
    Image.new("RGB", (50, 50), "red").save(img_buf, format="PNG")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("01.png", img_buf.getvalue())
        zf.writestr("readme.txt", b"not an image")
    src = buf.getvalue()

    out = izip2webpzip.convert(src, on_error="skip")
    entries = _entries(out)

    assert "01.webp" in entries
    assert "readme.txt" not in entries


def test_on_error_raise_propagates() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("readme.txt", b"not an image")
    src = buf.getvalue()

    with pytest.raises(Exception):
        izip2webpzip.convert(src, on_error="raise")


def test_log_func_called() -> None:
    src = _make_zip({"01.png": Image.new("RGB", (50, 50), "red")})
    seen: list[str] = []

    izip2webpzip.convert(src, log_func=seen.append)
    assert "01.png" in seen


def test_transform_applied() -> None:
    src = _make_zip({"01.png": Image.new("RGB", (200, 100), "red")})

    def shrink(img: Image.Image) -> Image.Image:
        return img.resize((50, 25))

    out = izip2webpzip.convert(src, transform=shrink)
    with Image.open(io.BytesIO(_entries(out)["01.webp"])) as decoded:
        assert decoded.size == (50, 25)


def test_to_webp_bytes_pure_function() -> None:
    img_bytes = io.BytesIO()
    Image.new("RGB", (50, 50), "red").save(img_bytes, format="PNG")

    out = izip2webpzip.to_webp_bytes(img_bytes.getvalue(), quality=80)
    assert out[:4] == b"RIFF"
    assert b"WEBP" in out[:12]


def test_la_mode_image_via_transform() -> None:
    """LA mode (grayscale + alpha) is not directly WebP-savable in some Pillow
    versions, so users handle it via transform."""
    src = _make_zip({"01.png": Image.new("LA", (50, 50), (128, 200))})

    def flatten(img: Image.Image) -> Image.Image:
        if img.mode in ("LA", "RGBA"):
            return img.convert("RGB")
        return img

    out = izip2webpzip.convert(src, transform=flatten)
    assert _entries(out)["01.webp"][:4] == b"RIFF"
