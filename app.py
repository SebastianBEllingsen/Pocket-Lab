from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from collections import Counter
from yolo_detection import detect_cells, draw_cells, WBC_TYPES
import cv2
import os

app = Flask(__name__)

# Folder where uploaded images will be saved (absolute, so the app
# works no matter which directory it is started from)
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")

# Create the folder if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_image = None
current_cells = []  # cells found by YOLO in current_image
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
    global current_image, current_cells, zoom_level

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

    # Save the uploaded image (secure_filename blocks paths like "../../x")
    filename = secure_filename(file.filename)

    if filename == "":
        return jsonify({
            "success": False,
            "message": "Invalid file name"
        })

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Read the image with OpenCV
    current_image = cv2.imread(filepath)

    # Check if OpenCV could read the image
    if current_image is None:
        return jsonify({
            "success": False,
            "message": "Could not read image with OpenCV"
        })

    # New image starts at 100% zoom
    zoom_level = 1.0

    # Find the cells once; the count/show buttons reuse the result
    current_cells = detect_cells(current_image)

    # Print information about the image
    print("Image opened successfully!")
    print("Image shape:", current_image.shape)

    # Send image location back to JavaScript
    return jsonify({
        "success": True,
        "message": "Image opened successfully",
        "image_url": url_for("static", filename="uploads/" + filename)
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

def save_result_image(image, filename):
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, filename), image)
    return url_for("static", filename="uploads/" + filename)


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
    image_url = save_result_image(zoomed_image, "zoomed_image.jpg")
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

    count = sum(1 for cell in current_cells if cell["type"] == "rbc")

    return jsonify({
        "success": True,
        "count": count,
        "message": f"Found {count} red blood cells"
    })


@app.route("/count-white", methods=["POST"])
def count_white():
    if current_image is None:
        return jsonify({"success": False, "count": "--", "message": "No image loaded"})

    types = Counter(cell["type"] for cell in current_cells if cell["type"] in WBC_TYPES)
    count = sum(types.values())
    details = ", ".join(f"{n} {name}" for name, n in types.items())

    return jsonify({
        "success": True,
        "count": count,
        "message": f"Found {count} white blood cells" + (f" ({details})" if details else "")
    })


# -------------------------
# SHOW CELLS
# -------------------------

def show_cells(filename, message, **options):
    if current_image is None:
        return jsonify({"success": False, "message": "No image loaded"})

    result = draw_cells(current_image, current_cells, **options)
    image_url = save_result_image(result, filename)

    return jsonify({
        "success": True,
        "message": message,
        "image_url": image_url
    })


@app.route("/show-red", methods=["POST"])
def show_red():
    return show_cells("cells_red.jpg", "Showing red blood cells", show_red=True, show_white=False)


@app.route("/show-white", methods=["POST"])
def show_white():
    return show_cells("cells_white.jpg", "Showing white blood cells", show_red=False, show_white=True)


@app.route("/show-all", methods=["POST"])
def show_all():
    return show_cells("cells_all.jpg", "Showing all cells", show_platelets=True)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
