
// ------------------------------------
// Send button action to Flask
// ------------------------------------
async function sendAction(url) {

    try {

        const response = await fetch(url, {
            method: "POST"
        });

        const data = await response.json();

        if (data.success) {

            // If Flask sends a new image,
            // update the image in the editor
            if (data.image_url) {

                const image =
                    document.getElementById("editorImage");

                image.src =
                    data.image_url + "?t=" + new Date().getTime();

                image.style.display = "block";
            }

            // Update zoom percentage
            if (data.zoom) {

                document.getElementById(
                    "zoom-level"
                ).textContent = data.zoom + "%";
            }
        }

        showNotification(data.message);

        console.log(data);

    } catch (error) {

        console.error(error);

        showNotification("Something went wrong.");

    }
}


// ------------------------------------
// Count cells
// ------------------------------------

async function countCells(url, resultElement) {

    try {

        const response = await fetch(url, {
            method: "POST"
        });

        const data = await response.json();

        document.getElementById(resultElement).textContent =
            data.count;

        showNotification(data.message);

    } catch (error) {

        console.error(error);

        showNotification("Counting failed.");

    }
}


// ------------------------------------
// Notification
// ------------------------------------

function showNotification(message) {

    const notification =
        document.getElementById("notification");

    notification.textContent = message;

    notification.classList.add("show");

    setTimeout(() => {

        notification.classList.remove("show");

    }, 2500);
}



function toggleShapes() {
    const menu = document.getElementById("shapesMenu");

    if (menu.classList.contains("show")) {
        menu.classList.remove("show");
    } else {
        menu.classList.add("show");
    }
}

function toggleMobileTools() {

    const menu = document.getElementById("mobileToolsMenu");

    menu.classList.toggle("show");

}


function selectTool(tool) {

    console.log("Selected tool:", tool);

    // Deactivate Zoom when another tool is selected
    deactivateZoom();

    // Close the tools menu
    document.getElementById("mobileToolsMenu")
        .classList.remove("show");


    if (tool === "crop") {
        console.log("Crop selected");
    }

    else if (tool === "draw") {
        console.log("Draw selected");
    }

    else if (tool === "text") {
        startText();
    }

    else if (tool === "arrow") {
        console.log("Arrow selected");
    }

    else if (tool === "rectangle") {
        console.log("Rectangle selected");
    }

    else if (tool === "circle") {
        console.log("Circle selected");
    }

}