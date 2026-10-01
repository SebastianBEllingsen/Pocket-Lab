from flask import Flask, render_template, request, jsonify, url_for
from cell_detection import detect_cells, draw_cells
from processing.zoom import zoom_in
from processing.image import open_image
from processing.crop import crop_image
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")

# Create the folder if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_image = None
original_image = None
zoom_level = 1.0

# -------------------------
# Main page
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/open-image", methods=["POST"])
def open_image_route():
    global current_image, original_image, zoom_level

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
    zoom_level = 1.0

    # Send image location back to JavaScript
    return jsonify({
        "success": True,
        "message": "Image opened successfully",
        "image_url": url_for(
            "static",
            filename="uploads/" + filename
        )
    })

@app.route("/save-image", methods=["POST"])
def save_image():
    # TODO: Add OpenCV save code here
    return jsonify({
        "success": True,
        "message": "Save Image clicked - OpenCV code goes here"
    })


@app.route("/reset", methods=["POST"])
def reset():
    # TODO: Restore original image using OpenCV
    return jsonify({
        "success": True,
        "message": "Reset clicked - OpenCV code goes here"
    })


# -------------------------
# TOOLS
# -------------------------

@app.route("/rectangle", methods=["POST"])
def rectangle():
    # TODO: Add cv2.rectangle()
    return jsonify({
        "success": True,
        "message": "Rectangle tool selected"
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

    current_image = crop_image(
        current_image,
        x,
        y,
        width,
        height
    )

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
    # TODO: Add cv2.putText()
    return jsonify({
        "success": True,
        "message": "Text tool selected"
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

    # Save zoomed image
    image_url = save_result_image(zoomed_image, "zoomed_image.jpg")
    return jsonify({
        "success": True,
        "message": "Zoom in",
        "image_url": image_url
    })



@app.route("/zoom-out", methods=["POST"])
def zoom_out():
    global current_image, zoom_level

    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    # Decrease zoom level
    zoom_level = max(0.2, zoom_level - 0.2)

    height, width = current_image.shape[:2]

    new_width = int(width * zoom_level)
    new_height = int(height * zoom_level)

    zoomed_image = cv2.resize(
        current_image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Save zoomed image
    image_url = save_result_image(zoomed_image, "zoomed_image.jpg")
    return jsonify({
        "success": True,
        "message": "Zoom out",
        "zoom": int(zoom_level * 100),
        "image_url": image_url
    })


# -------------------------
# ADJUSTMENTS
# -------------------------

@app.route("/greyscale", methods=["POST"])
def greyscale():
    # TODO:
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return jsonify({
        "success": True,
        "message": "Greyscale selected"
    })


# -------------------------
# UNDO / REDO
# -------------------------

@app.route("/undo", methods=["POST"])
def undo():
    # TODO: Restore previous image from history
    return jsonify({
        "success": True,
        "message": "Undo"
    })


@app.route("/redo", methods=["POST"])
def redo():
    # TODO: Restore next image from history
    return jsonify({
        "success": True,
        "message": "Redo"
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
