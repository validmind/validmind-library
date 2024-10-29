# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

import base64
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DOWNSAMPLE_PERCENTAGE = 50


def open_base64_image(base64_string):
    if base64_string.startswith("data:image/png;base64,"):
        base64_string = base64_string.split(",")[1]

    image_data = base64.b64decode(base64_string)
    image_buffer = io.BytesIO(image_data)
    image = Image.open(image_buffer)

    return image


def downsample_image(base64_string):
    image = open_base64_image(base64_string)

    # Calculate the target dimensions based on the reduction percentage
    target_width = int(image.width * (1 - DOWNSAMPLE_PERCENTAGE / 100))
    target_height = int(image.height * (1 - DOWNSAMPLE_PERCENTAGE / 100))

    # If the image is already smaller than the target size, return the original
    if image.width <= target_width and image.height <= target_height:
        return base64_string

    # remove any margins from the image
    # Find the bounding box of non-uniform pixels (margin detection)
    width, height = image.size
    background = image.getpixel((0, 0))  # Assume top-left pixel is background color

    def is_different(pixel):
        return pixel != background

    left = next(
        x
        for x in range(width)
        if any(is_different(image.getpixel((x, y))) for y in range(height))
    )
    right = next(
        x
        for x in range(width - 1, -1, -1)
        if any(is_different(image.getpixel((x, y))) for y in range(height))
    )
    top = next(
        y
        for y in range(height)
        if any(is_different(image.getpixel((x, y))) for x in range(width))
    )
    bottom = next(
        y
        for y in range(height - 1, -1, -1)
        if any(is_different(image.getpixel((x, y))) for x in range(width))
    )

    # Crop the image to remove the uniform margin (with some padding)
    bbox = (left - 5, top - 5, right + 6, bottom + 6)
    image = image.crop(bbox)

    # If the image has an alpha channel, remove any transparent margins
    if image.mode in ("RGBA", "LA"):
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        if bbox:
            image = image.crop(bbox)

    # Apply unsharp mask to enhance edges
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # Calculate new dimensions
    aspect_ratio = image.width / image.height
    new_height = target_height
    new_width = int(new_height * aspect_ratio)

    # print(f"downsampling from {width}x{height} to {new_width}x{new_height}")

    # Ensure we don't exceed the target width
    if new_width > target_width:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

    # print(f"downsampling from {image.width}x{image.height} to {new_width}x{new_height}")

    # Convert to numpy array for custom downsampling
    img_array = np.array(image)

    # Optimized area interpolation
    h_factor = img_array.shape[0] / new_height
    w_factor = img_array.shape[1] / new_width

    h_indices = (np.arange(new_height).reshape(-1, 1) * h_factor).astype(int)
    w_indices = (np.arange(new_width).reshape(1, -1) * w_factor).astype(int)

    h_indices = np.minimum(h_indices, img_array.shape[0] - 1)
    w_indices = np.minimum(w_indices, img_array.shape[1] - 1)

    # Convert back to PIL Image
    image = Image.fromarray(img_array[h_indices, w_indices].astype(np.uint8))

    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)

    # Sharpen the image
    image = image.filter(ImageFilter.SHARPEN)

    # Convert the image to bytes in PNG format
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Encode the bytes to base64
    b64_encoded = base64.b64encode(img_bytes).decode("utf-8")

    return f"data:image/png;base64,{b64_encoded}"
