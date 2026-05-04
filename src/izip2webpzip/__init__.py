from importlib.metadata import PackageNotFoundError, version

from izip2webpzip.izip2webpzip import (
    configure_pillow_safety,
    convert,
    to_webp_bytes,
)

try:
    __version__ = version("izip2webpzip")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["convert", "to_webp_bytes", "configure_pillow_safety", "__version__"]
