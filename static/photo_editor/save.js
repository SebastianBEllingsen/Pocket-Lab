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

        // Check if the browser supports Save As
        if ("showSaveFilePicker" in window) {

            // Open the PC Save As dialog
            const fileHandle =
                await window.showSaveFilePicker({

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

        } else {

            // Fallback for browsers that do not support Save As
            const imageUrl =
                URL.createObjectURL(imageBlob);

            const link =
                document.createElement("a");

            link.href = imageUrl;
            link.download = "PocketLab_result.jpg";

            document.body.appendChild(link);

            link.click();

            document.body.removeChild(link);

            URL.revokeObjectURL(imageUrl);
        }


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