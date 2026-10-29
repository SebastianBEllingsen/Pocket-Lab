import cv2


def add_text(image, text, x, y, size, color):

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Line width grows with the text size, so text stays readable on large photos
    thickness = max(1, round(size * 1.5))

    # Maximum width available for the text
    max_width = image.shape[1] - x - 10

    # Split text into words
    words = text.split()

    lines = []
    current_line = ""

    # Build lines that fit inside the image
    for word in words:

        test_line = (
            current_line + " " + word
        ).strip()

        (text_width, text_height), _ = cv2.getTextSize(
            test_line,
            font,
            size,
            thickness
        )

        if text_width <= max_width:
            current_line = test_line

        else:
            if current_line:
                lines.append(current_line)

            current_line = word

    # Add the last line
    if current_line:
        lines.append(current_line)


    # Draw each line
    line_height = int(40 * size)

    for i, line in enumerate(lines):
        (text_width, text_height), baseline = cv2.getTextSize(
            line,
            font,
            size,
            thickness
        )

        line_y = y + text_height + (i * line_height)

        cv2.putText(
            image,
            line,
            (x, line_y),
            font,
            size,
            color,
            thickness
        )

    return image