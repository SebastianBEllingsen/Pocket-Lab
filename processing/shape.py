import cv2


def draw_rectangle(
        image,
        x,
        y,
        width,
        height,
        color,
        thickness
):
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        color,
        thickness
    )

    return image


def draw_circle(
        image,
        center_x,
        center_y,
        radius,
        color,
        thickness
):
    cv2.circle(
        image,
        (center_x, center_y),
        radius,
        color,
        thickness
    )

    return image


def draw_arrow(
        image,
        x1,
        y1,
        x2,
        y2,
        color,
        thickness
):
    cv2.arrowedLine(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    return image