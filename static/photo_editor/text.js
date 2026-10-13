let textX = 0;
let textY = 0;
let textModeActive = false;

function hexToBgr(hex) {

    const r = parseInt(hex.substring(1, 3), 16);
    const g = parseInt(hex.substring(3, 5), 16);
    const b = parseInt(hex.substring(5, 7), 16);

    return [b, g, r];
}


function startText() {

    const image = document.getElementById("editorImage");

    if (image.style.display === "none") {
        showNotification("Open an image first");
        return;
    }

    if (!activateTool("text")) {
        return;
    }

    // Activate text mode
    textModeActive = true;

    image.style.cursor = "text";

    showNotification("Click on the image to add text");

    image.addEventListener("click", textImageClick, { once: true }
    );
}




function textImageClick(event) {

    const image =
        document.getElementById("editorImage");

    const textBox =
        document.getElementById("textInputBox");

    const textInput =
        document.getElementById("textInput");

    image.removeEventListener("click", textImageClick);

    textModeActive = false;

    const rect =
        image.getBoundingClientRect();

    textX = event.clientX - rect.left;
    textY = event.clientY - rect.top;


    textBox.style.left = textX + "px";
    textBox.style.top = textY + "px";

   textBox.style.display = "block";

    document
        .getElementById("textControls")
        .style.display = "flex";

   updateTextInputStyle();
    updateTextBoxSize();

    textInput.focus();
}


// ------------------------------------
// Move Text Box
// ------------------------------------

let isMovingText = false;
let textStartX = 0;
let textStartY = 0;

let isResizingText = false;
let resizeStartX = 0;
let resizeStartWidth = 0;

const textBox = document.getElementById("textInputBox");
const textResizeHandle = document.getElementById("textResizeHandle");


textBox.addEventListener("mousedown", function(event) {

     const rect = textBox.getBoundingClientRect();

    // Remember where inside the text box
    // the user clicked
    textStartX = event.clientX - rect.left;
    textStartY = event.clientY - rect.top;

    isMovingText = true;


    event.preventDefault();
    textResizeHandle.addEventListener(
    "mousedown",
    function(event) {

        isResizingText = true;

        resizeStartX =
            event.clientX;

        resizeStartWidth =
            textBox.offsetWidth;

        event.preventDefault();

        event.stopPropagation();
    }
);
});


document.addEventListener("mousemove", function(event) {

     if (isResizingText) {

        const wrapper =
            document.querySelector(".image-wrapper");

        const wrapperRect =
            wrapper.getBoundingClientRect();


        const newWidth =
            resizeStartWidth +
            (event.clientX - resizeStartX);


        const maxWidth =
            wrapperRect.right -
            textBox.getBoundingClientRect().left;


        const width =
            Math.max(
                80,
                Math.min(
                    newWidth,
                    maxWidth
                )
            );


        textBox.style.width =
            width + "px";


        return;
    }
    if (!isMovingText) {
        return;
    }

    const wrapper =
        document.querySelector(".image-wrapper");

    const wrapperRect =
        wrapper.getBoundingClientRect();

    let newX =
        event.clientX -
        wrapperRect.left -
        textStartX;

    let newY =
        event.clientY -
        wrapperRect.top -
        textStartY;


    const maxX =
    wrapperRect.width - textBox.offsetWidth;

    const maxY =
        wrapperRect.height - textBox.offsetHeight;


    newX = Math.max(
        0,
        Math.min(newX, maxX)
    );

    newY = Math.max(
        0,
        Math.min(newY, maxY)
    );


    textBox.style.left =
        newX + "px";

    textBox.style.top =
        newY + "px";

    textX = newX;
    textY = newY;

});


document.addEventListener("mouseup", function() {

    isMovingText = false;
    isResizingText = false;
});

// ------------------------------------
// Apply Text
// ------------------------------------

async function applyText() {

    const image =
        document.getElementById("editorImage");

    const textBox =
        document.getElementById("textInputBox");

    const textInput =
        document.getElementById("textInput");

    const text =
        textInput.value.trim();

    const textSize =
    parseFloat(
        document.getElementById("textSize").value
    );

    const textColor =
    document.getElementById("textColor").value;

    const color =
    hexToBgr(textColor);

    if (text === "") {
        showNotification("Enter some text first");
        return;
    }

    // Get the displayed image size
    const rect =
        image.getBoundingClientRect();

    // Convert displayed coordinates
    // to the actual image coordinates
    const scaleX =
        image.naturalWidth / rect.width;

    const scaleY =
        image.naturalHeight / rect.height;

    const x =
        Math.round(textX * scaleX);

    const y =
        Math.round(textY * scaleY);


    try {

        const response = await fetch("/text", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text,
                x: x,
                y: y,
                size: textSize,
                color: color
            })
        });

        const data = await response.json();

        if (data.success) {

            textModeActive = false;
            deactivateTool("text");

            image.removeEventListener("click", textImageClick);
            image.src =
                data.image_url +
                "?t=" +
                new Date().getTime();

            image.style.display = "block";

            textBox.style.display = "none";

            document
                .getElementById("textControls")
                .style.display = "none";


            image.style.cursor = "default";

            textInput.value = "";

            showNotification(data.message);

        } else {

            showNotification(data.message);

        }

    } catch (error) {

        console.error(error);

        showNotification("Failed to add text.");

    }
}


// ------------------------------------
// Cancel Text
// ------------------------------------

function cancelText() {

    const image =
        document.getElementById("editorImage");

    const textBox =
        document.getElementById("textInputBox");

    const textInput =
        document.getElementById("textInput");

    // Deactivate text mode
    textModeActive = false;
    deactivateTool("text");

    // Remove click listener
    image.removeEventListener("click", textImageClick);


    // Clear the input
    textInput.value = "";

    // Hide the text input box
    textBox.style.display = "none";

    document
        .getElementById("textControls")
        .style.display = "none";


    // Reset cursor
    document.getElementById("editorImage").style.cursor = "default";

    showNotification("Text cancelled");
}

function updateTextInputStyle() {

    const textInput =
        document.getElementById("textInput");

    const textSize =
        parseFloat(
            document.getElementById("textSize").value
        );

    const textColor =
        document.getElementById("textColor").value;


    // Change text size
    textInput.style.fontSize =
        (textSize * 20) + "px";


    // Change text color
    textInput.style.color =
        textColor;
}

function updateTextBoxSize() {

    const textInput =
        document.getElementById("textInput");

    const textBox =
        document.getElementById("textInputBox");


    // Reset height so the textarea can shrink
    textInput.style.height = "auto";


    // Set height based on the content
    textInput.style.height =
        textInput.scrollHeight + "px";


    // Set width based on the text
    const text =
        textInput.value;

    if (text === "") {

        textBox.style.width = "200px";

        return;
    }


    const canvas =
        document.createElement("canvas");

    const context =
        canvas.getContext("2d");


    context.font =
        getComputedStyle(textInput).font;


    const lines =
        text.split("\n");


    let maxWidth = 0;


    for (const line of lines) {

        const width =
            context.measureText(line).width;

        maxWidth =
            Math.max(
                maxWidth,
                width
            );
    }


    const padding =
        20;


    const newWidth =
        maxWidth + padding;


    textBox.style.width =
        Math.max(
            80,
            newWidth
        ) + "px";
}


document
    .getElementById("textSize")
    .addEventListener(
        "input",
        function() {

            updateTextInputStyle();

            updateTextBoxSize();

        }
    );


document
    .getElementById("textColor")
    .addEventListener(
        "input",
        updateTextInputStyle
    );

document
    .getElementById("textInput")
    .addEventListener(
        "input",
        updateTextBoxSize
    );