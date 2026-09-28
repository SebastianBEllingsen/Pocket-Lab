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


async function openImage() {

    const input = document.getElementById("imageInput");

    // Åpne PC-filvelgeren
    input.click();

    // Vent til brukeren velger et bilde
    input.onchange = async function () {

        const file = input.files[0];

        if (!file) {
            return;
        }

        // Lag en FormData
        const formData = new FormData();

        formData.append("image", file);

        // Send bildet til Flask
        const response = await fetch("/open-image", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

if (data.success) {

    // Get the image element
    const image =
        document.getElementById("editorImage");

    // Show the selected image
    image.src = data.image_url;
    image.style.display = "block";


    // Hide the microscope placeholder
    document.getElementById(
        "imagePlaceholder"
    ).style.display = "none";


    // Reset zoom display for the new image
    document.getElementById(
        "zoom-level"
    ).textContent = "100%";


    // Update image name
    document.getElementById(
        "image-status"
    ).textContent = file.name;


    showNotification(
        "Image opened successfully"
    );

}else {

            alert(data.message);
        }
    };
}

function toggleShapes() {

    const menu = document.getElementById("shapesMenu");

    menu.classList.toggle("show");
}

