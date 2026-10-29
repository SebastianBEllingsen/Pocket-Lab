
// ------------------------------------
// Send button action to Flask
// ------------------------------------
async function sendAction(url) {

    // Showing cells runs the cell analysis, which can take a few seconds
    const analysing = url.startsWith("/show-");

    if (analysing) {
        setBusy(true);
    }

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

    } finally {

        if (analysing) {
            setBusy(false);
        }
    }
}


// ------------------------------------
// Busy state while the cell analysis runs
// ------------------------------------

function setBusy(busy) {

    // Disable the analysis buttons so the analysis is not started twice
    document.querySelectorAll(".analysis-button").forEach((button) => {
        button.disabled = busy;
    });

    if (busy) {
        showNotification("Analysing cells...", 0);
    }
}


// ------------------------------------
// Analysis results box
// ------------------------------------

const RESULTS_TEXT = "Image analysis results will appear here.";

function showResults(data) {

    const lines = [];

    if (data.types) {
        lines.push("White cell types: " + data.types + ".");
    }

    lines.push(
        "Cells are counted inside the red circle. White cell types need " +
        "100x and can only tell lymphocytes from neutrophils."
    );

    document.getElementById("analysis-results").textContent =
        lines.join(" ");
}


// ------------------------------------
// Clear old counts when the image changes
// ------------------------------------

const COUNT_IDS = [
    "red-count",
    "white-count",
    "platelet-count",
    "smudge-count"
];

function clearCounts() {

    COUNT_IDS.forEach((id) => {
        document.getElementById(id).textContent = "--";
    });

    document.getElementById("analysis-results").textContent =
        RESULTS_TEXT;
}

// Every edit (open, crop, text, shape, zoom, undo, redo, reset) shows a
// new image. Only the "Show ... Cells" images (cells_*.jpg) belong to the
// current counts, so any other new image clears them.
new MutationObserver(() => {

    const src =
        document.getElementById("editorImage").getAttribute("src") || "";

    if (!src.includes("/uploads/cells_")) {
        clearCounts();
    }

}).observe(document.getElementById("editorImage"), {
    attributes: true,
    attributeFilter: ["src"]
});


// ------------------------------------
// Count cells
// ------------------------------------

async function countCells(url, resultElement) {

    setBusy(true);

    try {

        const response = await fetch(url, {
            method: "POST"
        });

        const data = await response.json();

        document.getElementById(resultElement).textContent =
            data.count;

        // Count White also sends the white cell types
        if (data.types !== undefined) {
            showResults(data);
        }

        showNotification(data.message);

    } catch (error) {

        console.error(error);

        showNotification("Counting failed.");

    } finally {

        setBusy(false);
    }
}


// ------------------------------------
// Count all cell types at once
// ------------------------------------

async function countAll() {

    setBusy(true);

    try {

        const response = await fetch("/count-all", {
            method: "POST"
        });

        const data = await response.json();

        if (data.success) {

            document.getElementById("red-count").textContent =
                data.counts.rbc;

            document.getElementById("white-count").textContent =
                data.counts.wbc;

            document.getElementById("platelet-count").textContent =
                data.counts.platelet;

            document.getElementById("smudge-count").textContent =
                data.counts.smudge;

            showResults(data);
        }

        showNotification(data.message);

    } catch (error) {

        console.error(error);

        showNotification("Counting failed.");

    } finally {

        setBusy(false);
    }
}


// ------------------------------------
// Notification
// ------------------------------------

let notificationTimer = null;

// duration 0 keeps the message until the next one (used while analysing)
function showNotification(message, duration = 2500) {

    const notification =
        document.getElementById("notification");

    notification.textContent = message;

    notification.classList.add("show");

    clearTimeout(notificationTimer);

    if (duration > 0) {

        notificationTimer = setTimeout(() => {

            notification.classList.remove("show");

        }, duration);
    }
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