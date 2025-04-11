import io
import os

import requests
from mcp.server.fastmcp import Image
from PIL import Image as PILImage
from PIL import ImageDraw


def retrieve_image_from_url(image_url: str) -> PILImage.Image:
    """
    Retrieve an image from a given URL and return it as a PIL Image object.
    """
    response = requests.get(image_url)
    response.raise_for_status()
    image_data = io.BytesIO(response.content)
    return PILImage.open(image_data)


def load_image(image: str | bytes | io.BufferedReader) -> PILImage.Image:
    """
    Load an image from a file path, URL, or raw bytes.

    Args:
        image: Image file path, URL, or raw bytes.

    Returns:
        PIL Image object.
    """
    if isinstance(image, io.BufferedReader):
        image = image.read()
        return PILImage.open(io.BytesIO(image))
    elif isinstance(image, bytes):
        return PILImage.open(io.BytesIO(image))
    elif os.path.isfile(image):
        return PILImage.open(image)
    elif image.startswith("http://") or image.startswith("https://"):
        return retrieve_image_from_url(image)
    else:
        raise ValueError(f"Invalid image path or URL: {image}")


def to_mcp_image(image: PILImage.Image | bytes, format: str = "jpeg") -> Image:
    """
    Convert a PIL Image object or bytes to an MCP Image.

    Args:
        image: PIL Image object or bytes containing image data
        format: Format to save the image in (default is "jpeg")

    Returns:
        MCP Image object with specified format
    """
    if isinstance(image, io.BufferedReader):
        image_bytes = image.read()
    elif isinstance(image, bytes):
        image_bytes = image
    elif isinstance(image, PILImage.Image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        image_bytes = img_byte_arr.getvalue()
    else:
        raise ValueError("Invalid image type. Expected PIL Image or bytes.")

    return Image(data=image_bytes, format=format)


def render_bounding_boxes(image: PILImage.Image, rois) -> PILImage.Image:
    """
    Draw bounding boxes on an image based on ROIs returned from a counting detector.

    Args:
        image: PIL Image object to draw on.
        rois: List of ROI objects returned from image_query.rois
    """
    width, height = image.size
    draw = ImageDraw.Draw(image)

    for roi in rois:
        x1 = int(roi.geometry.left * width)
        y1 = int(roi.geometry.top * height)
        x2 = int(roi.geometry.right * width)
        y2 = int(roi.geometry.bottom * height)

        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)

        label_text = f"{roi.label}: {roi.score:.2f}"
        draw.text((x1, y1 - 15), label_text, fill=(0, 255, 0))

    return image
