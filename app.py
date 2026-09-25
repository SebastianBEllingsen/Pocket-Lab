from flask import Flask, render_template, request, jsonify
import cv2
import os

app = Flask(__name__)

# Folder where uploaded images will be saved
UPLOAD_FOLDER = "static/uploads"

# Create the folder if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_image = None
zoom_level = 1.0

# -------------------------
# Main page
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------
# FILE MENU
# -------------------------

@app.route("/open-image", methods=["POST"])
def open_image():
    global current_image, zoom_level

    # Check if an image was uploaded
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image selected"
        })

    file = request.files["image"]

    # Check if a file was selected
    if file.filename == "":
        return jsonify({
            "success": False,
            "message": "No image selected"
        })

    # Save the uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Read the image with OpenCV
    current_image = cv2.imread(filepath)

    # Check if OpenCV could read the image
    if current_image is None:
        return jsonify({
            "success": False,
            "message": "Could not read image with OpenCV"
        })

    # Print information about the image
    print("Image opened successfully!")
    print("Image shape:", current_image.shape)

    # Send image location back to JavaScript
    return jsonify({
        "success": True,
        "message": "Image opened successfully",
        "image_url": "/" + filepath.replace("\\", "/")
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
    # TODO: Add OpenCV crop
    return jsonify({
        "success": True,
        "message": "Crop tool selected"
    })


@app.route("/text", methods=["POST"])
def text():
    # TODO: Add cv2.putText()
    return jsonify({
        "success": True,
        "message": "Text tool selected"
    })


# -------------------------
# ZOOM
# -------------------------

def save_zoomed_image(image):
    zoomed_path = os.path.join(
        UPLOAD_FOLDER,
        "zoomed_image.jpg"
    )

    cv2.imwrite(zoomed_path, image)

    return "/" + zoomed_path.replace("\\", "/")


@app.route("/zoom-in", methods=["POST"])

def zoom_in():
    global current_image, zoom_level

    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image loaded"
        })

    # Increase zoom level
    zoom_level += 0.2

    # Get original image size
    height, width = current_image.shape[:2]

    # Calculate new size
    new_width = int(width * zoom_level)
    new_height = int(height * zoom_level)

    # Resize image using OpenCV
    zoomed_image = cv2.resize(
        current_image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Save zoomed image
    image_url = save_zoomed_image(zoomed_image)

    return jsonify({
        "success": True,
        "message": "Zoom in",
        "zoom": int(zoom_level * 100),
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
    image_url = save_zoomed_image(zoomed_image)

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
    # TODO:
    # OpenCV segmentation + contour detection
    # Count red blood cells

    return jsonify({
        "success": True,
        "count": "--",
        "message": "Red cell counting will be implemented with OpenCV"
    })


@app.route("/count-white", methods=["POST"])
def count_white():
    # TODO:
    # OpenCV segmentation + contour detection
    # Count white blood cells

    return jsonify({
        "success": True,
        "count": "--",
        "message": "White cell counting will be implemented with OpenCV"
    })


# -------------------------
# SHOW CELLS
# -------------------------

@app.route("/show-red", methods=["POST"])
def show_red():
    # TODO: OpenCV segmentation
    return jsonify({
        "success": True,
        "message": "Show red cells - OpenCV code goes here"
    })


@app.route("/show-white", methods=["POST"])
def show_white():
    # TODO: OpenCV segmentation
    return jsonify({
        "success": True,
        "message": "Show white cells - OpenCV code goes here"
    })


@app.route("/show-all", methods=["POST"])
def show_all():
    # TODO: Show all detected cells
    return jsonify({
        "success": True,
        "message": "Show all cells - OpenCV code goes here"
    })


if __name__ == "__main__":
    app.run(debug=True)