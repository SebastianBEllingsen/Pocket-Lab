from werkzeug.utils import secure_filename
import cv2
import os


def open_image(file, upload_folder):
    if file is None or file.filename == "":
        return None, None, "No image selected"

    filename = secure_filename(file.filename)

    if filename == "":
        return None, None, "Invalid file name"

    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    image = cv2.imread(filepath)

    if image is None:
        return None, None, "Could not read image with OpenCV"

    print("Image opened successfully!")
    print("Image shape:", image.shape)

    return image, filename, None
