import cv2


def prepare_image_for_save(image):

    # Convert the OpenCV image to JPEG format
    success, buffer = cv2.imencode(".jpg", image)

    # Check if the image was encoded successfully
    if not success:
        return None

    return buffer.tobytes()