import math
from torchvision import transforms
from PIL import Image

MIN_PIXELS = 224 * 224

MAX_PIXELS = 4096 * 4096

MAX_RATIO = 40

def smart_resize(
    height: int, width: int, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    current_pixels = height * width
    if current_pixels > max_pixels:
        beta = math.sqrt(current_pixels / max_pixels)
        new_height = int(height / beta)
        new_width = int(width / beta)
    elif current_pixels < min_pixels:
        beta = math.sqrt(min_pixels / current_pixels)
        new_height = int(height * beta)
        new_width = int(width * beta)
    else:
        new_height = height
        new_width = width

    return new_height, new_width



def smart_resize_with_target(
    img: Image.Image,
    target_resolution: tuple[int, int],
    keep_aspect_ratio: bool = True,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Image.Image:
    original_width, original_height = img.size
    target_width, target_height = target_resolution

    # print(f"Original width: {original_width}, Target width: {target_width}")
    # print(f"Original height: {original_height}, Target height: {target_height}")
    if keep_aspect_ratio:
        # Calculate the scaling ratio while keeping aspect ratio
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

    else:
        # Directly use target resolution
        new_width, new_height = target_width, target_height


    # Apply smart_resize logic to ensure the resolution is within min_pixels and max_pixels
    h_bar, w_bar = smart_resize(new_height, new_width, min_pixels, max_pixels)
    return img.resize((w_bar, h_bar), Image.Resampling.LANCZOS)