let isDragging = false;
let activeHandle = null;

function startCrop() {

    const image =
        document.getElementById("editorImage");

    const overlay =
        document.getElementById("cropOverlay");

    if (image.style.display === "none") {
        showNotification("Open an image first");
        return;
    }

    // Try to activate Crop tool
    if (!activateTool("crop")) {
        return;
    }

    // Show crop overlay
    overlay.style.display = "block";

    overlay.style.left = "10%";
    overlay.style.top = "10%";
    overlay.style.width = "80%";
    overlay.style.height = "80%";

    document
        .querySelector(".crop-actions")
        .style.display = "flex";

    showNotification("Crop tool activated");
}


// Add mouse events to all four handles
document.querySelectorAll(".crop-handle").forEach(handle => {

    handle.addEventListener("mousedown", function(event) {

        isDragging = true;
        activeHandle = handle;

        event.preventDefault();
    });

});


// Move the active handle
document.addEventListener("mousemove", function(event) {

    if (!isDragging || !activeHandle) {
        return;
    }

    const overlay = document.getElementById("cropOverlay");
    const wrapper = document.querySelector(".image-wrapper");

    const wrapperRect = wrapper.getBoundingClientRect();

    const mouseX = event.clientX - wrapperRect.left;
    const mouseY = event.clientY - wrapperRect.top;

    const overlayLeft = overlay.offsetLeft;
    const overlayTop = overlay.offsetTop;
    const overlayRight = overlayLeft + overlay.offsetWidth;
    const overlayBottom = overlayTop + overlay.offsetHeight;

    const minSize = 30;


    // TOP
    if (activeHandle.classList.contains("top")) {

        let newTop = mouseY;

        newTop = Math.max(0, newTop);
        newTop = Math.min(newTop, overlayBottom - minSize);

        overlay.style.top =
            (newTop / wrapperRect.height) * 100 + "%";

        overlay.style.height =
            ((overlayBottom - newTop) / wrapperRect.height) * 100 + "%";
    }


    // BOTTOM
    if (activeHandle.classList.contains("bottom")) {

        let newBottom = mouseY;

        newBottom = Math.min(wrapperRect.height, newBottom);
        newBottom = Math.max(newBottom, overlayTop + minSize);

        overlay.style.height =
            ((newBottom - overlayTop) / wrapperRect.height) * 100 + "%";
    }


    // LEFT
    if (activeHandle.classList.contains("left")) {

        let newLeft = mouseX;

        newLeft = Math.max(0, newLeft);
        newLeft = Math.min(newLeft, overlayRight - minSize);

        overlay.style.left =
            (newLeft / wrapperRect.width) * 100 + "%";

        overlay.style.width =
            ((overlayRight - newLeft) / wrapperRect.width) * 100 + "%";
    }


    // RIGHT
    if (activeHandle.classList.contains("right")) {

        let newRight = mouseX;

        newRight = Math.min(wrapperRect.width, newRight);
        newRight = Math.max(newRight, overlayLeft + minSize);

        overlay.style.width =
            ((newRight - overlayLeft) / wrapperRect.width) * 100 + "%";
    }

});


// Stop dragging
document.addEventListener("mouseup", function() {

    isDragging = false;
    activeHandle = null;

});

async function applyCrop() {

    const image = document.getElementById("editorImage");
    const overlay = document.getElementById("cropOverlay");

    const imageRect = image.getBoundingClientRect();
    const overlayRect = overlay.getBoundingClientRect();

    // Crop coordinates on the displayed image
    const displayX = overlayRect.left - imageRect.left;
    const displayY = overlayRect.top - imageRect.top;

    const displayWidth = overlayRect.width;
    const displayHeight = overlayRect.height;

    // Convert displayed coordinates to original image coordinates
    const scaleX = image.naturalWidth / imageRect.width;
    const scaleY = image.naturalHeight / imageRect.height;

    const x = Math.round(displayX * scaleX);
    const y = Math.round(displayY * scaleY);
    const width = Math.round(displayWidth * scaleX);
    const height = Math.round(displayHeight * scaleY);


    // Send crop coordinates to Flask
    const response = await fetch("/crop", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            x: x,
            y: y,
            width: width,
            height: height
        })
    });

    const data = await response.json();

    if (data.success) {

        image.src = data.image_url + "?t=" + new Date().getTime();

        overlay.style.display = "none";
        document.querySelector(".crop-actions").style.display = "none";

        showNotification("Image cropped successfully");

    // Release Crop tool
    deactivateTool("crop");

    showNotification("Image cropped successfully");
    }
}

function cancelCrop() {

    const overlay = document.getElementById("cropOverlay");

    overlay.style.display = "none";

    document.querySelector(".crop-actions").style.display = "none";

    // Release Crop tool
    deactivateTool("crop");
    showNotification("Crop cancelled");
}