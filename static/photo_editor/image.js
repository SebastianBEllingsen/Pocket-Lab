

async function openImage() {

    const input = document.getElementById("imageInput");

    // Open the PC file picker
    input.click();

    // Wait for the user to select an image
    input.onchange = async function () {

        const file = input.files[0];

        if (!file) {
            return;
        }

        // Create a FormData object
        const formData = new FormData();

        formData.append("image", file);

        // Send the image to Flask
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

            // Reset the zoom display for the new image
            document.getElementById(
                "zoom-level"
            ).textContent = "100%";

            // Update the image name
const imageStatus =
    document.getElementById("image-status");

console.log("imageStatus:", imageStatus);

if (imageStatus) {
    imageStatus.textContent = file.name;
}

            showNotification(
                "Image opened successfully"
            );

        } else {

            alert(data.message);
        }
    };
}

