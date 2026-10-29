from flask import Flask, render_template, request, jsonify, url_for
from cell_analysis import analyse, draw
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
from collections import Counter

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")

# Create the folder if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_image = None
original_image = None

# Last image drawn by a "Show ... Cells" button (saved when it is on screen)
shown_image = None

# Stores all image versions
image_history = []

# Points to the currently selected image version
current_index = -1

zoom_level = 100.0

# -------------------------
# IMAGE HISTORY
# -------------------------

def add_to_history(image):
    global image_history, current_index, zoom_level

    # Remove redo history
    image_history = image_history[:current_index + 1]

    # Store image and current zoom level
    image_history.append({
        "image": image.copy(),
        "zoom": zoom_level
    })

    # Move to the newest history item
    current_index = len(image_history) - 1

    print("History length:", len(image_history))
    print("Current index:", current_index)
    print("Zoom:", zoom_level)

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

    # New image starts at 100% zoom
    zoom_level = 100.0

    # Start a new image history
    image_history = [{
        "image": image.copy(),
        "zoom": zoom_level
    }]
    current_index = 0


    # Send image location back to JavaScript
    return jsonify({
        "success": True,
        "message": "Image opened successfully",
        "image_url": url_for(
            "static",
            filename="uploads/" + filename
        ),
        "filename": filename,
        "zoom": zoom_level
    })

@app.route("/save-image", methods=["POST"])
def save_image():

    # Check if an image is loaded
    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    # Save what is on screen: the image with cell boxes if a
    # "Show ... Cells" result is displayed, otherwise the edited image
    data = request.get_json(silent=True) or {}

    if data.get("shown") and shown_image is not None:
        image = shown_image
    else:
        image = current_image

    # Prepare the image for saving
    prepared_image = prepare_image_for_save(image)

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

    global current_image, original_image, image_history, current_index, zoom_level

    # Check if an original image exists
    if original_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    # Restore the original image and zoom level
    current_image = original_image.copy()
    zoom_level = 100.0

    # Reset image history (same format as add_to_history, so Undo works)
    image_history = [{
        "image": current_image.copy(),
        "zoom": zoom_level
    }]
    current_index = 0

    # Save the reset image
    image_url = save_result_image(
        current_image,
        "reset_image.jpg"
    )

    return jsonify({
        "success": True,
        "message": "Image reset successfully",
        "image_url": image_url,
        "zoom": zoom_level
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

    # Keep the crop inside the image: the overlay can reach
    # a pixel past the edge, and negative values would wrap around
    image_height, image_width = current_image.shape[:2]

    x = max(0, int(data["x"]))
    y = max(0, int(data["y"]))
    width = min(int(data["width"]), image_width - x)
    height = min(int(data["height"]), image_height - y)

    if width < 2 or height < 2:
        return jsonify({
            "success": False,
            "message": "Crop area is too small"
        })

    print("Crop received:", x, y, width, height)

    # Create the cropped image (a copy, not a view of the history)
    cropped_image = crop_image(
        current_image,
        x,
        y,
        width,
        height
    ).copy()

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

    # Add text using OpenCV (on a copy, so the history entry stays unchanged)
    text_image = add_text(
        current_image.copy(),
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

    global current_image, zoom_level

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

    # zoom_in() shows half the width and height, so the view is 2x larger
    zoom_level *= 2

    # Add image and new zoom level to history
    add_to_history(zoomed_image)

    # Make the zoomed image the new current image
    current_image = zoomed_image

    image_url = save_result_image(
        zoomed_image,
        "zoomed_image.jpg"
    )

    return jsonify({
        "success": True,
        "message": "Zoom in",
        "image_url": image_url,
        "zoom": zoom_level
    })


@app.route("/zoom-out", methods=["POST"])
def zoom_out_route():

    global current_image, zoom_level

    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    # Go back to the newest earlier version with a lower zoom level,
    # i.e. the view before the last Zoom In
    for i in range(current_index - 1, -1, -1):
        if image_history[i]["zoom"] < zoom_level:
            break
    else:
        return jsonify({
            "success": False,
            "message": "Already zoomed out"
        })

    zoom_level = image_history[i]["zoom"]
    current_image = image_history[i]["image"].copy()

    # Zoom out is a new step, so Undo can return to the zoomed view
    add_to_history(current_image)

    image_url = save_result_image(
        current_image,
        "zoomed_out_image.jpg"
    )

    return jsonify({
        "success": True,
        "message": "Zoom out",
        "image_url": image_url,
        "zoom": zoom_level
    })



# -------------------------
# UNDO / REDO
# -------------------------

@app.route("/undo", methods=["POST"])
def undo():

    global current_image, current_index, zoom_level

    # Check if we can undo
    if current_index <= 0:
        return jsonify({
            "success": False,
            "message": "Nothing to undo"
        })

    # Move one step back in history
    current_index -= 1

    # Restore the previous image (a copy, so later edits cannot change the history)
    current_image = image_history[current_index]["image"].copy()
    zoom_level = image_history[current_index]["zoom"]

    # Save the restored image
    image_url = save_result_image(
        current_image,
        "undo_image.jpg"
    )

    return jsonify({
        "success": True,
        "message": "Undo",
        "image_url": image_url,
        "zoom": zoom_level
    })


@app.route("/redo", methods=["POST"])
def redo():

    global current_image, current_index, zoom_level
    if current_index >= len(image_history) - 1:
        return jsonify({
            "success": False,
            "message": "Nothing to redo"
        })

    current_index += 1

    # Restore image and zoom level (a copy, as in undo)
    current_image = image_history[current_index]["image"].copy()
    zoom_level = image_history[current_index]["zoom"]

    image_url = save_result_image(
        current_image,
        "redo_image.jpg"
    )

    return jsonify({
        "success": True,
        "message": "Redo",
        "image_url": image_url,
        "zoom": zoom_level
    })


# -------------------------
# CELL ANALYSIS
# -------------------------

def get_analysis():
    """Run (or reuse) the cell analysis of the current image.
    Returns (result, None) or (None, error response)."""
    if current_image is None:
        return None, jsonify({
            "success": False,
            "count": "--",
            "message": "No image loaded"
        })

    try:
        return analyse(current_image), None

    except Exception as error:
        # e.g. no round red cells found, so the cell size cannot be measured
        print("Analysis failed:", error)
        return None, jsonify({
            "success": False,
            "count": "--",
            "message": "Could not analyse this image (cell size not found)"
        })


# -------------------------
# CELL COUNTING
# -------------------------

def count_response(cell_class, label):
    result, error = get_analysis()
    if error is not None:
        return error

    count = result["counts"][cell_class]

    return jsonify({
        "success": True,
        "count": count,
        "message": f"Found {count} {label}"
    })


def type_summary(result):
    """White cell types of the image, e.g. '2 neutrophil, 1 lymphocyte'."""
    return ", ".join(
        f"{n} {t}" for t, n in Counter(result["wbc_types"]).most_common()
    )


@app.route("/count-red", methods=["POST"])
def count_red():
    return count_response("rbc", "red blood cells")


@app.route("/count-white", methods=["POST"])
def count_white():
    result, error = get_analysis()
    if error is not None:
        return error

    count = result["counts"]["wbc"]
    types = type_summary(result)

    return jsonify({
        "success": True,
        "count": count,
        "types": types,
        "message": f"Found {count} white blood cells" + (f": {types}" if types else "")
    })


@app.route("/count-all", methods=["POST"])
def count_all():
    result, error = get_analysis()
    if error is not None:
        return error

    counts = result["counts"]

    return jsonify({
        "success": True,
        "counts": counts,
        "types": type_summary(result),
        "message": (
            f"RBC {counts['rbc']}, WBC {counts['wbc']}, "
            f"platelets {counts['platelet']}, smudge {counts['smudge']}"
        )
    })


# -------------------------
# SHOW CELLS
# -------------------------

def show_response(classes, filename, message):
    global shown_image

    result, error = get_analysis()
    if error is not None:
        return error

    # Drawn on a copy: the boxes are not added to the image history,
    # but kept so that Save can store what is on screen
    shown_image = draw(current_image, result, classes)
    image_url = save_result_image(shown_image, filename)

    return jsonify({
        "success": True,
        "message": message,
        "image_url": image_url
    })


@app.route("/show-red", methods=["POST"])
def show_red():
    return show_response(["rbc"], "cells_red.jpg", "Showing red blood cells")


@app.route("/show-white", methods=["POST"])
def show_white():
    # smudge cells are shown with the white cells, so the two can be told apart
    return show_response(["wbc", "smudge"], "cells_white.jpg",
                         "Showing white blood cells (smudge cells in purple)")


@app.route("/show-all", methods=["POST"])
def show_all():
    return show_response(["rbc", "wbc", "platelet", "smudge"], "cells_all.jpg",
                         "Showing all cells")


if __name__ == "__main__":
    app.run(debug=True, port=5001)