from importlib.metadata import PackageNotFoundError, version

from izip2webpzip.izip2webpzip import (
    configure_pillow_safety,
    convert,
    convert_replace,
    to_webp_bytes,
    verify_zip_images,
)

try:
    __version__ = version("izip2webpzip")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "convert",
    "convert_replace",
    "to_webp_bytes",
    "verify_zip_images",
    "configure_pillow_safety",
    "__version__",
]
