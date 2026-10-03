async function saveImage() {

    try {

        // Ask Flask for the current image
        const response = await fetch("/save-image", {
            method: "POST"
        });

        // Check if Flask returned an error
        if (!response.ok) {
            showNotification("Could not save image.");
            return;
        }

        // Get the image as a Blob
        const imageBlob = await response.blob();

        // Open the PC Save As dialog
        const fileHandle = await window.showSaveFilePicker({

            suggestedName: "PocketLab_result.jpg",

            types: [
                {
                    description: "JPEG image",
                    accept: {
                        "image/jpeg": [".jpg"]
                    }
                }
            ]

        });

        // Create a writable file
        const writable =
            await fileHandle.createWritable();

        // Write the image to the selected file
        await writable.write(imageBlob);

        // Close the file
        await writable.close();

        showNotification("Image saved successfully.");

    } catch (error) {

        console.error(error);

        // User cancelled the Save As window
        if (error.name === "AbortError") {
            return;
        }

        showNotification("Something went wrong while saving.");

    }
}