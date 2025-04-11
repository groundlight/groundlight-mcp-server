import io
import os

import requests
from PIL import Image as PILImage


def retrieve_image_from_url(image_url: str) -> PILImage.Image:
    """
    Retrieve an image from a given URL and return it as a PIL Image object.
    """
    response = requests.get(image_url)
    response.raise_for_status()
    image_data = io.BytesIO(response.content)
    return PILImage.open(image_data)


def load_image(image: str | bytes) -> PILImage.Image:
    if isinstance(image, bytes):
        return PILImage.open(io.BytesIO(image))
    elif os.path.isfile(image):
        return PILImage.open(image)
    elif image.startswith("http://") or image.startswith("https://"):
        return retrieve_image_from_url(image)
    else:
        raise ValueError(f"Invalid image path or URL: {image}")
