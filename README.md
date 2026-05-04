izip2webpzip (image zip to WebP zip)
=====================================

Convert ZIP files containing images into ZIP files containing WebP images.

- Fast, in-memory conversion (no temp directory, HDD/SSD friendly).
- Drop-in replacement for the common manga `.cbz` recompression workflow.
- Composable: pass a `transform` callable to resize/rotate/alpha-flatten before encoding.

Supported input image formats: jpeg, jpeg2000, png, webp, avif, heif, psd, tiff, etc.

Usage
-----

    $ izip2webpzip sample1.zip
    $ izip2webpzip sample2.cbz sample3.cbz

Outputs are written next to each input as `sample1.webp.zip`, etc.

In Linux you can use shell glob:

    $ izip2webpzip sample*.zip

Installation
------------

    $ pip install izip2webpzip

Library
-------

```python
import izip2webpzip

# Path -> Path
izip2webpzip.convert("input.zip", "output.zip")

# bytes -> bytes (in-memory)
out_bytes = izip2webpzip.convert(open("input.zip", "rb").read())

# Lossless WebP (best quality, larger files)
izip2webpzip.convert("input.zip", "output.zip", lossless=True)

# With a custom image transform (resize, rotate, alpha-flatten, ...)
from PIL import Image, ImageOps

def my_transform(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img) or img
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img

izip2webpzip.convert("input.zip", "output.zip", transform=my_transform)

# Per-file error policy
izip2webpzip.convert("input.zip", on_error="keep")   # default: keep originals on failure
izip2webpzip.convert("input.zip", on_error="skip")   # drop failing entries
izip2webpzip.convert("input.zip", on_error="warn")   # like keep + warnings.warn
izip2webpzip.convert("input.zip", on_error="raise")  # stop on first failure

# Use the pure single-image helper
import izip2webpzip
webp_bytes = izip2webpzip.to_webp_bytes(image_bytes, quality=90, lossless=False)
```

### Security Options

For processing untrusted ZIP files, set safety limits:

```python
import izip2webpzip

# Default: Pillow's safe limits (~178 megapixels)
izip2webpzip.convert("untrusted.zip", "output.zip")

# Custom pixel cap
izip2webpzip.convert(
    "untrusted.zip", "output.zip",
    max_image_pixels=100_000_000,  # 100 megapixels
)

# Disable cap (UNSAFE; only for trusted input)
izip2webpzip.convert(
    "trusted.zip", "output.zip",
    max_image_pixels=0,
    load_truncated_images=True,
)
```

Design notes
------------

- **In-memory only**: source ZIP is read once into a `BytesIO`, output is built into another `BytesIO`. No `tempfile` extraction.
- **Pure functions where possible**: `to_webp_bytes()` is a side-effect-free helper; `configure_pillow_safety()` is a context manager that save/restores Pillow globals rather than mutating them outright.
- **Lazy plugin registration**: pillow-heif and pillow-avif have global side effects on import, so they're deferred to the first `convert()` call.
- **Composable transforms**: image-level operations (resize, EXIF rotate, alpha flatten) live in user-supplied `transform` callables rather than as kwargs, keeping the core API small.

Reference
---------

- Inspired by [izip2pdf](https://github.com/hy20191108/izip2pdf) (same author, ZIP→PDF variant).
