"""izip2webpzip: convert ZIP of images to ZIP of WebP images, in-memory."""

import argparse
import io
import sys
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Literal, Optional, Union

import tqdm  # type: ignore[import-untyped]
from natsort import natsorted  # type: ignore[import-untyped]
from PIL import Image, ImageFile, UnidentifiedImageError  # type: ignore[import-untyped]

# Pillow safety defaults (override via convert(max_image_pixels=...))
DEFAULT_MAX_IMAGE_PIXELS = 178956970  # Pillow's default ~178MP

# Extensions that Pillow can typically open as images. Anything else (e.g.
# ComicInfo.xml, thumbs.db) is copied into the output ZIP unchanged so that
# CBZ metadata is preserved.
IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".avif", ".heic", ".heif", ".psd", ".jp2",
})

ErrorMode = Literal["raise", "skip", "warn"]
ProgressMode = Union[bool, Literal["bar", "log"]]
Transform = Callable[[Image.Image], Image.Image]

_LOG_PREFIX = {"warn": "WARNING", "skip": "SKIPPED", "raise": "ERROR"}

_format_support_initialized = False


def _ensure_format_support() -> None:
    """Lazily register HEIF/AVIF Pillow plugins on first convert() call."""
    global _format_support_initialized
    if _format_support_initialized:
        return
    try:
        import pillow_avif  # type: ignore[import-untyped] # noqa: F401
        from pillow_heif import register_heif_opener  # type: ignore[import-untyped]
        register_heif_opener()
    except ImportError as e:
        raise ImportError(
            f"Optional format support unavailable: {e}. "
            "Install pillow-avif-plugin and pillow-heif to handle AVIF/HEIF input."
        ) from e
    _format_support_initialized = True


@contextmanager
def configure_pillow_safety(
    max_image_pixels: Optional[int] = None,
    load_truncated_images: bool = False,
) -> Iterator[None]:
    """Save/restore Pillow's process-wide safety settings.

    Pillow stores these as module attributes; this manager prevents callers
    from leaking changes to other code.

    Note: not thread-safe. In multi-threaded apps, set Pillow's globals once
    at startup instead of using this manager per-call.
    """
    saved_pixels = Image.MAX_IMAGE_PIXELS
    saved_truncated = ImageFile.LOAD_TRUNCATED_IMAGES

    if max_image_pixels is None:
        Image.MAX_IMAGE_PIXELS = DEFAULT_MAX_IMAGE_PIXELS
    elif max_image_pixels == 0:
        warnings.warn(
            "max_image_pixels=0 disables Pillow's decompression bomb guard. "
            "Only use with trusted input.",
            stacklevel=2,
        )
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

    Calls `_ensure_format_support()` to register HEIF/AVIF Pillow plugins on
    first invocation (idempotent). Wrap in `configure_pillow_safety(...)`
    when stricter Pillow safety limits are required.

    Note: WebP supports up to 16383 px per dimension. Larger images must be
    downscaled in `transform=` first; otherwise Pillow raises OSError.
    """
    _ensure_format_support()
    output = io.BytesIO()
    with Image.open(io.BytesIO(image_bytes)) as image:
        if transform is not None:
            image = transform(image)
        image.save(output, "webp", quality=quality, lossless=lossless)
    return output.getvalue()


def _is_image_entry(name: str) -> bool:
    """Whether the entry's extension matches a supported image format."""
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def _load_zip_bytes(zip_input: Union[str, bytes, Path]) -> bytes:
    """Return raw ZIP bytes; accepts a file path or bytes directly."""
    if isinstance(zip_input, bytes):
        return zip_input
    return Path(zip_input).read_bytes()


def _emit_failure(
    message: str,
    on_error: ErrorMode,
    log_func: Optional[Callable[[str], None]],
) -> None:
    """Notify a per-image failure according to on_error policy.

    The "raise" path is handled in convert() before reaching here.
    """
    if on_error == "warn" or log_func is None:
        warnings.warn(message, stacklevel=3)
    if log_func is not None:
        prefix = _LOG_PREFIX.get(on_error, on_error.upper())
        log_func(f"{prefix}: {message}")


def convert(
    zip_input: Union[str, bytes, Path],
    output_zip_path: Optional[Union[str, Path]] = None,
    progress: ProgressMode = False,
    *,
    log_func: Optional[Callable[[str], None]] = None,
    quality: int = 85,
    lossless: bool = False,
    transform: Optional[Transform] = None,
    on_error: ErrorMode = "warn",
    max_image_pixels: Optional[int] = None,
    load_truncated_images: bool = False,
) -> bytes:
    """Convert a ZIP of images into a ZIP of WebP images, entirely in memory.

    Non-image entries (judged by file extension, e.g. ComicInfo.xml) are
    always copied into the output ZIP unchanged. Image entries are converted
    to WebP; per-image conversion failures are handled per `on_error`.

    Args:
        zip_input: Path to a source ZIP, or its bytes.
        output_zip_path: If given, write the result there as well as returning it.
        progress: False/True/"bar"/"log" tqdm progress display modes.
        log_func: Receives every filename and warning text.
        quality: WebP quality 0-100. Ignored if lossless=True.
        lossless: Lossless WebP encoding (larger files, exact pixels).
        transform: Optional PIL.Image -> PIL.Image hook (resize/rotate/etc).
        on_error: per-image conversion failure policy.
            "raise" -> stop and propagate the exception.
            "skip"  -> drop the failing image entry from the output ZIP.
            "warn"  -> copy original bytes into the output ZIP and emit a
                       warnings.warn(...) (default).
            Note: transform bugs and unexpected exceptions (MemoryError, etc.)
            always propagate regardless of `on_error`.
        max_image_pixels: Pillow decompression-bomb guard.
            None -> Pillow's safe default (~178MP).
            0    -> disabled (UNSAFE; emits warning).
        load_truncated_images: Allow truncated images.

    Returns:
        ZIP bytes containing converted WebP images and pass-through entries.
    """
    _ensure_format_support()
    src_buf = io.BytesIO(_load_zip_bytes(zip_input))
    out_buf = io.BytesIO()

    show_bar = progress in (True, "bar", "log")
    show_log = progress == "log"

    with (
        configure_pillow_safety(max_image_pixels, load_truncated_images),
        zipfile.ZipFile(src_buf) as src_zip,
        zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as out_zip,
    ):
        names = list(natsorted(src_zip.namelist()))
        names_iter = tqdm.tqdm(names, desc="izip2webpzip") if show_bar else names

        for name in names_iter:
            if zipfile.Path(src_zip, name).is_dir():
                continue
            if show_log:
                tqdm.tqdm.write(name)
            if log_func is not None:
                log_func(name)

            data = src_zip.read(name)

            # Non-image entries always pass through unchanged so that CBZ
            # metadata (ComicInfo.xml, etc.) survives the conversion.
            if not _is_image_entry(name):
                out_zip.writestr(name, data)
                continue

            try:
                webp_data = to_webp_bytes(
                    data, quality=quality, lossless=lossless, transform=transform
                )
            except (UnidentifiedImageError, OSError, SyntaxError) as e:
                # Known Pillow image-decode/encode failures.
                if on_error == "raise":
                    raise
                if on_error == "warn":
                    out_zip.writestr(name, data)  # keep original bytes
                _emit_failure(f"Failed to convert {name}: {e}", on_error, log_func)
            else:
                webp_name = str(Path(name).with_suffix(".webp"))
                out_zip.writestr(webp_name, webp_data)
            # Other exceptions (transform bugs, MemoryError) propagate up.

    result = out_buf.getvalue()
    if output_zip_path is not None:
        Path(output_zip_path).write_bytes(result)
    return result


def main() -> None:
    """CLI: izip2webpzip [--quality N] [--lossless] [--no-progress] <zip>...

    Each input <name>.zip is converted to <name>.webp.zip, skipped if that
    output already exists. Returns the count of failed inputs as exit code.
    """
    parser = argparse.ArgumentParser(
        description="Convert ZIP of images to ZIP of WebP images, in-memory.",
    )
    parser.add_argument("zip_files", nargs="+", help="Input ZIP/CBZ files")
    parser.add_argument(
        "-q", "--quality", type=int, default=85,
        help="WebP quality 0-100 (default: 85, ignored with --lossless)",
    )
    parser.add_argument(
        "--lossless", action="store_true",
        help="Lossless WebP encoding (larger files, exact pixels)",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable per-file progress bar",
    )
    args = parser.parse_args()

    show_progress: ProgressMode = False if args.no_progress else True
    error_count = 0
    for zippath in tqdm.tqdm(args.zip_files, desc="process"):
        tqdm.tqdm.write(zippath)
        out_path = Path(zippath).with_suffix(".webp.zip")
        if out_path.is_file():
            continue  # idempotent: skip already-converted files
        try:
            convert(
                zippath, out_path, progress=show_progress,
                quality=args.quality, lossless=args.lossless,
                on_error="warn", log_func=tqdm.tqdm.write,
            )
        except Exception as e:
            tqdm.tqdm.write(f"ERROR: {zippath}: {e}", file=sys.stderr)
            error_count += 1
    sys.exit(error_count)


if __name__ == "__main__":
    main()
