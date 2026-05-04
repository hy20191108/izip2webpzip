"""izip2webpzip: convert ZIP of images to ZIP of WebP images, in-memory."""

import io
import sys
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Literal, Optional, Union

import tqdm  # type: ignore[import-untyped]
from natsort import natsorted  # type: ignore[import-untyped]
from PIL import Image, ImageFile  # type: ignore[import-untyped]

# Pillow safety defaults (override via convert(max_image_pixels=...))
DEFAULT_MAX_IMAGE_PIXELS = 178956970  # Pillow's default ~178MP

# WebP hard limit per dimension is 16383; we cap one step below to leave
# margin for off-by-one issues in some encoders.
WEBP_MAX_DIMENSION = 16383

ErrorMode = Literal["raise", "skip", "keep", "warn"]
Transform = Callable[[Image.Image], Image.Image]

_format_support_initialized = False


def _ensure_format_support() -> None:
    """Lazily register HEIF and AVIF Pillow plugins.

    These have global side effects (modifying Pillow's format registry),
    so we defer registration until the first convert() call rather than
    importing them at module load time.
    """
    global _format_support_initialized
    if _format_support_initialized:
        return
    import pillow_avif  # type: ignore[import-untyped] # noqa: F401
    from pillow_heif import register_heif_opener  # type: ignore[import-untyped]
    register_heif_opener()
    _format_support_initialized = True


@contextmanager
def configure_pillow_safety(
    max_image_pixels: Optional[int] = None,
    load_truncated_images: bool = False,
) -> Iterator[None]:
    """Temporarily configure Pillow's process-wide safety settings.

    Pillow stores these as module attributes, so this manager saves and
    restores the previous values to avoid leaking changes across calls.

    Note: not thread-safe. In multi-threaded apps, set Pillow's globals
    once at startup instead of using this manager.
    """
    saved_pixels = Image.MAX_IMAGE_PIXELS
    saved_truncated = ImageFile.LOAD_TRUNCATED_IMAGES

    if max_image_pixels is None:
        Image.MAX_IMAGE_PIXELS = DEFAULT_MAX_IMAGE_PIXELS
    elif max_image_pixels == 0:
        Image.MAX_IMAGE_PIXELS = None
    else:
        Image.MAX_IMAGE_PIXELS = max_image_pixels
    ImageFile.LOAD_TRUNCATED_IMAGES = load_truncated_images

    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = saved_pixels
        ImageFile.LOAD_TRUNCATED_IMAGES = saved_truncated


def to_webp_bytes(
    image_bytes: bytes,
    *,
    quality: int = 85,
    lossless: bool = False,
    transform: Optional[Transform] = None,
) -> bytes:
    """Convert one image's bytes into WebP bytes.

    A pure function: no global state, no I/O, no Pillow safety mutation.
    Wrap the call in `configure_pillow_safety(...)` if needed.

    Args:
        image_bytes: Raw bytes of an image (any Pillow-readable format).
        quality: WebP quality 0-100. Ignored if lossless=True.
        lossless: True for lossless WebP. Best for crisp art / pixel-perfect.
        transform: Optional PIL.Image -> PIL.Image hook applied before save.
                   Use this for resize, rotate, alpha-flatten, etc.
    """
    output = io.BytesIO()
    with Image.open(io.BytesIO(image_bytes)) as image:
        if transform is not None:
            image = transform(image)
        image.save(output, "webp", quality=quality, lossless=lossless)
    return output.getvalue()


def _read_zip_input(zip_input: Union[str, bytes]) -> bytes:
    if isinstance(zip_input, bytes):
        return zip_input
    with open(zip_input, "rb") as f:
        return f.read()


def _emit(
    message: str,
    on_error: ErrorMode,
    log_func: Optional[Callable[[str], None]],
) -> None:
    """Emit a per-file error message according to on_error policy."""
    if on_error == "warn":
        warnings.warn(message)
    if log_func is not None:
        log_func(f"WARNING: {message}" if on_error == "warn"
                 else f"SKIPPED: {message}" if on_error == "skip"
                 else f"KEPT: {message}")


def convert(
    zip_input: Union[str, bytes],
    output_zip_path: Optional[str] = None,
    progress: bool = False,
    *,
    log_func: Optional[Callable[[str], None]] = None,
    quality: int = 85,
    lossless: bool = False,
    transform: Optional[Transform] = None,
    on_error: ErrorMode = "keep",
    max_image_pixels: Optional[int] = None,
    load_truncated_images: bool = False,
) -> bytes:
    """Convert a ZIP of images into a ZIP of WebP images, entirely in memory.

    Args:
        zip_input: Path to a source ZIP, or its bytes.
        output_zip_path: If given, write the result there as well as returning it.
        progress: True for a tqdm progress bar.
        log_func: Receives every filename and warning text.
        quality: WebP quality 0-100. Ignored if lossless=True.
        lossless: Lossless WebP output (larger files, exact pixels).
        transform: Optional PIL.Image -> PIL.Image hook (resize/rotate/etc).
        on_error: per-image failure policy.
            "raise" -> stop and propagate the exception
            "skip"  -> drop the failing entry from the output ZIP
            "keep"  -> copy original bytes into the output ZIP unchanged (default)
            "warn"  -> like "keep" but also emits a warnings.warn(...)
        max_image_pixels: Decompression-bomb guard.
            None -> Pillow default (~178MP), 0 -> disabled (UNSAFE).
        load_truncated_images: Allow truncated images.

    Returns:
        ZIP file as bytes containing converted WebP images. Non-image entries
        and entries that failed to convert are handled per `on_error`.
    """
    _ensure_format_support()
    src_bytes = _read_zip_input(zip_input)

    src_buf = io.BytesIO(src_bytes)
    dst_buf = io.BytesIO()

    with configure_pillow_safety(max_image_pixels, load_truncated_images), \
         zipfile.ZipFile(src_buf) as src, \
         zipfile.ZipFile(dst_buf, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        names = list(natsorted(src.namelist()))
        names_iter = tqdm.tqdm(names, desc="izip2webpzip") if progress else names

        for name in names_iter:
            if zipfile.Path(src, name).is_dir():
                continue
            if log_func is not None:
                log_func(name)

            data = src.read(name)
            try:
                webp_data = to_webp_bytes(
                    data, quality=quality, lossless=lossless, transform=transform
                )
                webp_name = str(Path(name).with_suffix(".webp"))
                dst.writestr(webp_name, webp_data)
            except Exception as e:
                if on_error == "raise":
                    raise
                if on_error in ("keep", "warn"):
                    dst.writestr(name, data)
                _emit(f"Failed to convert {name}: {e}", on_error, log_func)

    return dst_buf.getvalue() if output_zip_path is None else _write_and_return(
        dst_buf.getvalue(), output_zip_path
    )


def _write_and_return(data: bytes, path: str) -> bytes:
    with open(path, "wb") as f:
        f.write(data)
    return data


def main() -> None:
    """CLI: izip2webpzip <zip>... — convert each ZIP to <name>.webp.zip."""
    for zippath in tqdm.tqdm(sys.argv[1:], desc="process"):
        tqdm.tqdm.write(zippath)
        out_path = Path(zippath).with_suffix(".webp.zip")
        if out_path.is_file():
            continue
        convert(zippath, str(out_path), progress=True)


if __name__ == "__main__":
    main()
