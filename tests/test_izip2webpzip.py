"""Tests for izip2webpzip."""
import io
import zipfile

import pytest
from PIL import Image, UnidentifiedImageError

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


@pytest.fixture
def mixed_zip() -> bytes:
    """ZIP with one PNG image and one non-image (text) entry."""
    buf = io.BytesIO()
    img_buf = io.BytesIO()
    Image.new("RGB", (50, 50), "red").save(img_buf, format="PNG")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("01.png", img_buf.getvalue())
        zf.writestr("ComicInfo.xml", b"<?xml version='1.0'?><ComicInfo />")
    return buf.getvalue()


@pytest.fixture
def corrupted_image_zip() -> bytes:
    """ZIP with one valid PNG and one corrupted entry that has an image extension."""
    buf = io.BytesIO()
    img_buf = io.BytesIO()
    Image.new("RGB", (50, 50), "red").save(img_buf, format="PNG")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("01.png", img_buf.getvalue())
        zf.writestr("broken.png", b"this is not a real PNG")
    return buf.getvalue()


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


def test_pathlib_path_accepted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src_path = tmp_path / "in.zip"
    src_path.write_bytes(_make_zip({"a.png": Image.new("RGB", (50, 50), "blue")}))
    out_path = tmp_path / "out.zip"

    out = izip2webpzip.convert(src_path, out_path)
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
    assert "subdir/" not in entries


def test_non_image_passes_through(mixed_zip: bytes) -> None:
    """Non-image extensions (e.g. xml) should be copied as-is regardless of on_error."""
    out = izip2webpzip.convert(mixed_zip, on_error="skip")  # even with skip
    entries = _entries(out)

    assert "01.webp" in entries
    assert "ComicInfo.xml" in entries
    assert entries["ComicInfo.xml"] == b"<?xml version='1.0'?><ComicInfo />"


def test_on_error_warn_keeps_original_with_warning(corrupted_image_zip: bytes) -> None:
    """warn: corrupted image entry kept as-is + warnings.warn emitted."""
    with pytest.warns(UserWarning, match="Failed to convert broken.png"):
        out = izip2webpzip.convert(corrupted_image_zip, on_error="warn")

    entries = _entries(out)
    assert "01.webp" in entries
    assert "broken.png" in entries  # kept as original


def test_on_error_skip_drops_failing(corrupted_image_zip: bytes) -> None:
    """skip: corrupted image entry dropped from output."""
    out = izip2webpzip.convert(corrupted_image_zip, on_error="skip", log_func=lambda _: None)
    entries = _entries(out)
    assert "01.webp" in entries
    assert "broken.png" not in entries


def test_on_error_raise_propagates(corrupted_image_zip: bytes) -> None:
    """raise: corrupted image triggers UnidentifiedImageError or OSError."""
    with pytest.raises((UnidentifiedImageError, OSError)):
        izip2webpzip.convert(corrupted_image_zip, on_error="raise")


def test_transform_exception_propagates() -> None:
    """transform-side bugs are NOT swallowed by on_error."""
    src = _make_zip({"01.png": Image.new("RGB", (50, 50), "red")})

    def buggy_transform(img):  # type: ignore[no-untyped-def]
        raise AttributeError("simulated transform bug")

    with pytest.raises(AttributeError, match="simulated transform bug"):
        izip2webpzip.convert(src, transform=buggy_transform, on_error="warn")


def test_log_func_called(mixed_zip: bytes) -> None:
    """log_func receives every entry name."""
    seen: list[str] = []
    izip2webpzip.convert(mixed_zip, log_func=seen.append)
    assert "01.png" in seen
    assert "ComicInfo.xml" in seen


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


def test_max_image_pixels_zero_warns() -> None:
    """max_image_pixels=0 disables decompression-bomb guard; must warn."""
    src = _make_zip({"01.png": Image.new("RGB", (50, 50), "red")})

    with pytest.warns(UserWarning, match="decompression bomb guard"):
        izip2webpzip.convert(src, max_image_pixels=0)


def test_la_mode_image_via_transform() -> None:
    """LA mode (grayscale + alpha) is handled by user via transform."""
    src = _make_zip({"01.png": Image.new("LA", (50, 50), (128, 200))})

    def flatten(img: Image.Image) -> Image.Image:
        if img.mode in ("LA", "RGBA"):
            return img.convert("RGB")
        return img

    out = izip2webpzip.convert(src, transform=flatten)
    assert _entries(out)["01.webp"][:4] == b"RIFF"


def test_version_exposed() -> None:
    assert hasattr(izip2webpzip, "__version__")
    assert isinstance(izip2webpzip.__version__, str)


def test_verify_zip_images_passes_for_valid_output() -> None:
    src = _make_zip({
        "01.png": Image.new("RGB", (50, 50), "red"),
        "02.png": Image.new("RGB", (60, 60), "green"),
    })
    out = izip2webpzip.convert(src)

    ok, failures = izip2webpzip.verify_zip_images(out)
    assert ok
    assert failures == []


def test_verify_zip_images_detects_corruption() -> None:
    # A ZIP whose .png entry is just garbage bytes.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("broken.png", b"this is definitely not PNG data")
    corrupted = buf.getvalue()

    ok, failures = izip2webpzip.verify_zip_images(corrupted)
    assert not ok
    assert any("broken.png" in f for f in failures)


def test_verify_zip_images_skips_non_image_entries(mixed_zip: bytes) -> None:
    """ComicInfo.xml etc. are not verified (since they're not images)."""
    # mixed_zip contains "01.png" + "ComicInfo.xml". Verify still passes since
    # only the image is checked and it's valid.
    ok, failures = izip2webpzip.verify_zip_images(mixed_zip)
    assert ok, failures


def test_verify_zip_images_sample_size_subset() -> None:
    images = {f"{i:02d}.png": Image.new("RGB", (10, 10), "red") for i in range(20)}
    src = _make_zip(images)

    ok, failures = izip2webpzip.verify_zip_images(src, sample_size=5)
    assert ok
    assert failures == []


def test_convert_replace_overwrites_source(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "manga.cbz"
    src.write_bytes(_make_zip({"01.png": Image.new("RGB", (50, 50), "red")}))
    original_size = src.stat().st_size

    izip2webpzip.convert_replace(src)

    assert src.exists()
    # File replaced; size differs from original (WebP is smaller for this content).
    assert src.stat().st_size != original_size
    assert "01.webp" in _entries(src.read_bytes())
    # Temp file cleaned up.
    assert not (tmp_path / "manga.cbz.izip2webpzip.tmp").exists()


def test_convert_replace_keeps_source_on_verify_failure(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch,
) -> None:
    src = tmp_path / "manga.cbz"
    original_bytes = _make_zip({"01.png": Image.new("RGB", (50, 50), "red")})
    src.write_bytes(original_bytes)

    # Force verification to report failure (patch the actual module, not
    # the re-export in __init__, so convert_replace's local reference picks
    # up the patched function).
    from izip2webpzip import izip2webpzip as core
    monkeypatch.setattr(
        core, "verify_zip_images",
        lambda *a, **kw: (False, ["fake_failure: simulated"]),
    )

    with pytest.raises(ValueError, match="Verification failed"):
        izip2webpzip.convert_replace(src)

    # Source unchanged, temp removed.
    assert src.read_bytes() == original_bytes
    assert not (tmp_path / "manga.cbz.izip2webpzip.tmp").exists()


def test_convert_replace_cleans_temp_on_convert_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "manga.cbz"
    original_bytes = _make_zip({"01.png": Image.new("RGB", (50, 50), "red")})
    src.write_bytes(original_bytes)

    def boom(_img):  # type: ignore[no-untyped-def]
        raise RuntimeError("transform exploded")

    with pytest.raises(RuntimeError, match="transform exploded"):
        izip2webpzip.convert_replace(src, transform=boom)

    assert src.read_bytes() == original_bytes
    assert not (tmp_path / "manga.cbz.izip2webpzip.tmp").exists()
