from werkzeug.utils import secure_filename
import cv2
import os


def open_image(file, upload_folder):
    # Check if a file was selected
    if file is None or file.filename == "":
        return None, None, "No image selected"

    # Make the filename safe
    filename = secure_filename(file.filename)

    if filename == "":
        return None, None, "Invalid file name"

    # Create the full file path
    filepath = os.path.join(upload_folder, filename)

    # Save the uploaded image
    file.save(filepath)

    # Read the image with OpenCV
    image = cv2.imread(filepath)

    # Check if OpenCV could read the image
    if image is None:
        return None, None, "Could not read image with OpenCV"

    # Print information about the image
    print("Image opened successfully!")
    print("Image shape:", image.shape)

    return image, filename, None