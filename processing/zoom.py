import cv2


def zoom_in(image, x, y):

    # Get the image dimensions
    height, width = image.shape[:2]

    # Size of the area we want to zoom into
    crop_width = width // 2
    crop_height = height // 2

    # Calculate crop boundaries around the clicked point
    x1 = max(0, x - crop_width // 2)
    y1 = max(0, y - crop_height // 2)

    x2 = min(width, x1 + crop_width)
    y2 = min(height, y1 + crop_height)

    # Crop the selected area
    cropped = image[y1:y2, x1:x2]

    # Resize the cropped area back to the original size
    zoomed = cv2.resize(
        cropped,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return zoomed