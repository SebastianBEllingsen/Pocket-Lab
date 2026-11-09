import cv2


def prepare_image_for_save(image):

    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        return None

    return buffer.tobytes()
