let zoomMode = false;

// ------------------------------------
// Deactivate Zoom In
// ------------------------------------
function deactivateZoom() {

    zoomMode = false;

    console.log("Zoom In mode deactivated");
}

// ------------------------------------
// Activate Zoom In
// ------------------------------------
function activateZoomIn() {

    zoomMode = true;

    console.log("Zoom In mode activated");
}

// ------------------------------------
// Detect click on image
// ------------------------------------
const image = document.getElementById("editorImage");

image.addEventListener("click", async function (event) {

    if (!zoomMode) {
        return;
    }

    console.log("Image clicked!");

    // Get the displayed image position
    const rect = image.getBoundingClientRect();
    console.log("Image rectangle:", rect);
    console.log("Natural size:", image.naturalWidth, image.naturalHeight);
    console.log("Displayed size:", rect.width, rect.height);

    // Get click position inside the displayed image
    const displayX = event.clientX - rect.left;
    const displayY = event.clientY - rect.top;

    // Calculate the scale between displayed image and original image
    const scaleX = image.naturalWidth / rect.width;
    const scaleY = image.naturalHeight / rect.height;

    // Convert displayed coordinates to original image coordinates
    const x = displayX * scaleX;
    const y = displayY * scaleY;

    console.log("Displayed position:", displayX, displayY);
    console.log("Original image position:", x, y);
    console.log("Natural image size:", image.naturalWidth, image.naturalHeight);
    console.log("Displayed image size:", rect.width, rect.height);



    // Send the click position to Flask
    const response = await fetch("/zoom-in", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            x: x,
            y: y
        })
    });

    const data = await response.json();

    console.log("Flask response:", data);

    // Display the zoomed image
    if (data.success) {

        image.src =
            data.image_url + "?t=" + new Date().getTime();

        console.log("Zoomed image displayed");

        // Update zoom percentage
        if (data.zoom) {

            document.getElementById(
                "zoom-level"
            ).textContent = data.zoom + "%";
        }

    } else {

        console.log(data.message);
    }

});