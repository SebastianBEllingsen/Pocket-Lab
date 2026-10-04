from flask import Flask, render_template, request, jsonify, url_for
from cell_detection import detect_cells, draw_cells
from processing.zoom import zoom_in
from processing.image import open_image
from processing.crop import crop_image
from processing.save import prepare_image_for_save
from processing.text import add_text
from processing.shape import (
    draw_rectangle,
    draw_circle,
    draw_arrow
)
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")

# Create the folder if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_image = None
original_image = None

# Stores all image versions
image_history = []

# Points to the currently selected image version
current_index = -1

zoom_level = 1.0

# -------------------------
# IMAGE HISTORY
# -------------------------

def add_to_history(image):
    global image_history, current_index

    # Remove redo history
    image_history = image_history[:current_index + 1]

    # Store a copy of the image
    image_history.append(image.copy())

    # Move to the newest image
    current_index = len(image_history) - 1

    print("History length:", len(image_history))
    print("Current index:", current_index)

# -------------------------
# Main page
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/open-image", methods=["POST"])
def open_image_route():
    global current_image, original_image, image_history, current_index, zoom_level

    # Check if an image was uploaded
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image selected"
        })

    file = request.files["image"]

    # Open the image using the function from processing/image.py
    image, filename, error = open_image(
        file,
        UPLOAD_FOLDER
    )

    # Check if opening the image failed
    if error is not None:
        return jsonify({
            "success": False,
            "message": error
        })

    # Store the opened image
    current_image = image
    original_image = current_image.copy()

    # Start a new image history
    image_history = [image.copy()]
    current_index = 0

    # New image starts at 100% zoom
    zoom_level = 1.0

    # Send image location back to JavaScript
    return jsonify({
        "success": True,
        "message": "Image opened successfully",
        "image_url": url_for(
            "static",
            filename="uploads/" + filename
        ),
        "filename": filename
    })

@app.route("/save-image", methods=["POST"])
def save_image():

    # Check if an image is loaded
    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    # Prepare the current image for saving
    prepared_image = prepare_image_for_save(current_image)

    # Check if the image could be prepared
    if prepared_image is None:
        return jsonify({
            "success": False,
            "message": "Could not prepare image for saving"
        })

    # Return the image data
    return prepared_image, 200, {
        "Content-Type": "image/jpeg"
    }


@app.route("/reset", methods=["POST"])
def reset():

    global current_image, original_image, image_history, current_index

    # Check if an original image exists
    if original_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    # Restore the original image
    current_image = original_image.copy()

    # Reset image history
    image_history = [current_image.copy()]
    current_index = 0

    # Save the reset image
    image_url = save_result_image(
        current_image,
        "reset_image.jpg"
    )

    return jsonify({
        "success": True,
        "message": "Image reset successfully",
        "image_url": image_url
    })



# -------------------------
# TOOLS
# -------------------------
@app.route("/shape", methods=["POST"])
def shape():

    global current_image

    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    data = request.get_json()

    shape_type = data["shape"]

    color = data.get(
        "color",
        [0, 0, 255]
    )

    thickness = int(
        data.get("thickness", 2)
    )


    # Rectangle
    if shape_type == "rectangle":

        x = int(data["x"])
        y = int(data["y"])
        width = int(data["width"])
        height = int(data["height"])

        result = draw_rectangle(
            current_image.copy(),
            x,
            y,
            width,
            height,
            color,
            thickness
        )


    # Circle
    elif shape_type == "circle":

        center_x = int(
            data["center_x"]
        )

        center_y = int(
            data["center_y"]
        )

        radius = int(
            data["radius"]
        )

        result = draw_circle(
            current_image.copy(),
            center_x,
            center_y,
            radius,
            color,
            thickness
        )


    # Arrow
    elif shape_type == "arrow":

        x1 = int(data["x1"])
        y1 = int(data["y1"])
        x2 = int(data["x2"])
        y2 = int(data["y2"])

        result = draw_arrow(
            current_image.copy(),
            x1,
            y1,
            x2,
            y2,
            color,
            thickness
        )


    else:

        return jsonify({
            "success": False,
            "message": "Unknown shape"
        })


    # Add the completed shape
    # as one history entry
    add_to_history(result)

    current_image = result


    image_url = save_result_image(
        current_image,
        "shape_image.jpg"
    )


    return jsonify({
        "success": True,
        "message": (
            shape_type.capitalize()
            + " added successfully"
        ),
        "image_url": image_url
    })


@app.route("/contour", methods=["POST"])
def contour():
    # TODO: Add free-hand contour OpenCV logic
    return jsonify({
        "success": True,
        "message": "Contour tool selected"
    })


@app.route("/crop", methods=["POST"])
def crop():

    global current_image

    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    data = request.get_json()

    x = data["x"]
    y = data["y"]
    width = data["width"]
    height = data["height"]

    print("Crop received:")
    print("x:", x)
    print("y:", y)
    print("width:", width)
    print("height:", height)

    # Create the cropped image
    cropped_image = crop_image(
        current_image,
        x,
        y,
        width,
        height
    )

    # Add the cropped image to history
    add_to_history(cropped_image)

    # Make the cropped image the current image
    current_image = cropped_image

    filename = "cropped_image.jpg"

    image_url = save_result_image(
        current_image,
        filename
    )

    return jsonify({
        "success": True,
        "message": "Image cropped successfully",
        "image_url": image_url
    })


@app.route("/text", methods=["POST"])
def text():

    global current_image

    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    data = request.get_json()

    text_value = data["text"]
    x = int(data["x"])
    y = int(data["y"])

    size = float(data["size"])
    color = data["color"]

    # Add text using OpenCV
    text_image = add_text(
        current_image,
        text_value,
        x,
        y,
        size,
        color
    )

    # Add the new image to history
    add_to_history(text_image)

    # Make the text image the current image
    current_image = text_image

    # Save the result
    image_url = save_result_image(
        current_image,
        "text_image.jpg"
    )

    return jsonify({
        "success": True,
        "message": "Text added successfully",
        "image_url": image_url
    })


def save_result_image(image, filename):
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, filename), image)
    return url_for("static", filename="uploads/" + filename)


@app.route("/zoom-in", methods=["POST"])
def zoom_in_route():
    global current_image

    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    # Get click position from JavaScript
    data = request.get_json()

    x = int(data["x"])
    y = int(data["y"])

    print("Zoom point:", x, y)

    # Perform zoom using the function from zoom.py
    zoomed_image = zoom_in(current_image, x, y)

    # Add the new zoomed image to history
    add_to_history(zoomed_image)

    # Make the zoomed image the new current image
    current_image = zoomed_image

    # Save zoomed image
    image_url = save_result_image(zoomed_image, "zoomed_image.jpg")
    return jsonify({
        "success": True,
        "message": "Zoom in",
        "image_url": image_url
    })



# -------------------------
# UNDO / REDO
# -------------------------

@app.route("/undo", methods=["POST"])
def undo():

    global current_image, current_index

    # Check if we can undo
    if current_index <= 0:
        return jsonify({
            "success": False,
            "message": "Nothing to undo"
        })

    # Move one step back in history
    current_index -= 1

    # Restore the previous image
    current_image = image_history[current_index]

    # Save the restored image
    image_url = save_result_image(
        current_image,
        "undo_image.jpg"
    )

    return jsonify({
        "success": True,
        "message": "Undo",
        "image_url": image_url

                   })


@app.route("/redo", methods=["POST"])
def redo():
    global current_image, current_index

    # Check if we can redo
    if current_index >= len(image_history) - 1:
        return jsonify({
            "success": False,
            "message": "Nothing to redo"
        })

    # Move one step forward in history
    current_index += 1

    # Restore the next image
    current_image = image_history[current_index]

    # Save the restored image
    image_url = save_result_image(
        current_image,
        "redo_image.jpg"
    )

    return jsonify({
        "success": True,
        "message": "Redo",
        "image_url": image_url
                   })

# -------------------------
# CELL COUNTING
# -------------------------

@app.route("/count-red", methods=["POST"])
def count_red():
    if current_image is None:
        return jsonify({"success": False, "count": "--", "message": "No image loaded"})

    rbc_contours, _ = detect_cells(current_image)

    return jsonify({
        "success": True,
        "count": len(rbc_contours),
        "message": f"Found {len(rbc_contours)} red blood cells"
    })


@app.route("/count-white", methods=["POST"])
def count_white():
    if current_image is None:
        return jsonify({"success": False, "count": "--", "message": "No image loaded"})

    _, wbc_contours = detect_cells(current_image)

    return jsonify({
        "success": True,
        "count": len(wbc_contours),
        "message": f"Found {len(wbc_contours)} white blood cells"
    })


# -------------------------¢
# SHOW CELLS
# -------------------------

@app.route("/show-red", methods=["POST"])
def show_red():
    if current_image is None:
        return jsonify({"success": False, "message": "No image loaded"})

    rbc_contours, wbc_contours = detect_cells(current_image)
    result = draw_cells(current_image, rbc_contours, wbc_contours, show_red=True, show_white=False)
    image_url = save_result_image(result, "cells_red.jpg")

    return jsonify({
        "success": True,
        "message": "Showing red blood cells",
        "image_url": image_url
    })


@app.route("/show-white", methods=["POST"])
def show_white():
    if current_image is None:
        return jsonify({"success": False, "message": "No image loaded"})

    rbc_contours, wbc_contours = detect_cells(current_image)
    result = draw_cells(current_image, rbc_contours, wbc_contours, show_red=False, show_white=True)
    image_url = save_result_image(result, "cells_white.jpg")

    return jsonify({
        "success": True,
        "message": "Showing white blood cells",
        "image_url": image_url
    })


@app.route("/show-all", methods=["POST"])
def show_all():
    if current_image is None:
        return jsonify({"success": False, "message": "No image loaded"})

    rbc_contours, wbc_contours = detect_cells(current_image)
    result = draw_cells(current_image, rbc_contours, wbc_contours)
    image_url = save_result_image(result, "cells_all.jpg")

    return jsonify({
        "success": True,
        "message": "Showing all cells",
        "image_url": image_url
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
