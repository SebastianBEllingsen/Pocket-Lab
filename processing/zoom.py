import cv2

def zoom_in(image, x, y):

    # Get the image dimensions
    height, width = image.shape[:2]

    # Size of the area we want to crop
    crop_width = width // 2
    crop_height = height // 2

    # Calculate half of the crop size
    half_width = crop_width // 2
    half_height = crop_height // 2

    # Calculate crop boundaries around the clicked point (to keep the click point in the center of the crop.)
    x1 = x - half_width
    x2 = x + half_width

    y1 = y - half_height
    y2 = y + half_height

    # Keep the crop inside the image
    if x1 < 0:
        x1 = 0
        x2 = crop_width

    if y1 < 0:
        y1 = 0
        y2 = crop_height

    if x2 > width:
        x2 = width
        x1 = width - crop_width

    if y2 > height:
        y2 = height
        y1 = height - crop_height

    # Crop the selected area
    cropped = image[y1:y2, x1:x2]

    # Resize the cropped area back to the original size
    zoomed = cv2.resize(
        cropped,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    return zoomed