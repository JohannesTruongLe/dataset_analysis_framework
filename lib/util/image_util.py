"""PIL Image util methods."""
import numpy as np
from PIL import Image


def save_cropped_image(source_path, output_path, x_min, x_max, y_min, y_max, zoom_factor=5):
    """Crop bounding box out of image.

    Since most of the time you also need context, the zoom factor make the crop out bigger.

    NOTE: If bounding box would reach out of image, the bounding box gets clipped.

    Args:
        source_path (str of pathlib.Path): Path of source image of size [x, y, channels]
        output_path (str of pathlib.Path): Path of where to save the crop out to
        x_min (int): Minimum X value of bounding box
        x_max (int): Maximum X value of bounding box
        y_min (int): Minimum Y value of bounding box
        y_max (int): Maximum Y value of bounding box
        zoom_factor (float or int):

    """
    image = np.array(Image.open(source_path))

    d_x = abs(x_min-x_max)/2 * zoom_factor
    d_y = abs(y_min-y_max)/2 * zoom_factor

    center_y = (y_max+y_min)/2
    center_x = (x_max+x_min)/2

    y_min = int(max(0, center_y - d_y))
    x_min = int(max(0, center_x - d_x))
    y_max = int(min(image.shape[0], center_y + d_y))
    x_max = int(min(image.shape[1], center_x + d_x))

    cropped_image = image[y_min:y_max, x_min:x_max]
    Image.fromarray(cropped_image).save(output_path)
