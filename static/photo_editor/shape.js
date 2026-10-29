let shapeMode = false;
let currentShape = null;

let startX = 0;
let startY = 0;

let endX = 0;
let endY = 0;

let isDrawingShape = false;


// Start a shape
function startShape(shape) {

    const image =
        document.getElementById("editorImage");

    if (image.style.display === "none") {
        showNotification("Open an image first");
        return;
    }

    // Try to activate Shape tool
    if (!activateTool("shape")) {
        return;
    }
    document.getElementById("shapesMenu").classList.remove("show");

    currentShape = shape;
    shapeMode = true;

    image.style.cursor = "crosshair";

    const canvas =
        document.getElementById("shapeCanvas");

    setupShapeCanvas();

    canvas.style.display = "block";

    showNotification(
        "Click and drag on the image to draw a " + shape
    );

    // Prevent duplicate event listeners
    image.removeEventListener(
        "mousedown",
        startShapeDrawing
    );

    image.addEventListener(
        "mousedown",
        startShapeDrawing
    );
}


// Start drawing
function startShapeDrawing(event) {

    if (!shapeMode) {
        return;
    }

    event.preventDefault();
    event.stopPropagation();

    const image =
        document.getElementById("editorImage");

    const rect =
        image.getBoundingClientRect();

    startX =
        event.clientX - rect.left;

    startY =
        event.clientY - rect.top;

    isDrawingShape = true;

    console.log(
        "Shape started:",
        currentShape,
        startX,
        startY
    );
}


// Draw preview while moving mouse
document.addEventListener(
    "mousemove",
    drawShapePreview
);


function drawShapePreview(event) {

    if (!isDrawingShape) {
        return;
    }

    const image =
        document.getElementById("editorImage");

    const canvas =
        document.getElementById("shapeCanvas");

    const rect =
        image.getBoundingClientRect();

    let currentX =
        event.clientX - rect.left;

    let currentY =
        event.clientY - rect.top;


    // Keep mouse inside image
    currentX = Math.max(
        0,
        Math.min(currentX, rect.width)
    );

    currentY = Math.max(
        0,
        Math.min(currentY, rect.height)
    );


    endX = currentX;
    endY = currentY;


    const context =
        canvas.getContext("2d");


    // Clear previous preview
    context.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    context.strokeStyle = "red";
    context.lineWidth = 2;


    // Rectangle
    if (currentShape === "rectangle") {

        const width =
            currentX - startX;

        const height =
            currentY - startY;

        context.strokeRect(
            startX,
            startY,
            width,
            height
        );
    }


    // Circle
    else if (currentShape === "circle") {

        const radius =
            Math.sqrt(
                Math.pow(currentX - startX, 2) +
                Math.pow(currentY - startY, 2)
            );

        context.beginPath();

        context.arc(
            startX,
            startY,
            radius,
            0,
            2 * Math.PI
        );

        context.stroke();
    }


    // Arrow
    else if (currentShape === "arrow") {

        drawArrowPreview(
            context,
            startX,
            startY,
            currentX,
            currentY
        );
    }
}


// Draw arrow preview
function drawArrowPreview(
    context,
    x1,
    y1,
    x2,
    y2
) {

    const headLength = 10;

    const angle =
        Math.atan2(
            y2 - y1,
            x2 - x1
        );


    // Main line
    context.beginPath();

    context.moveTo(x1, y1);

    context.lineTo(x2, y2);

    context.stroke();


    // Arrow head
    context.beginPath();

    context.moveTo(
        x2,
        y2
    );

    context.lineTo(
        x2 -
        headLength *
        Math.cos(angle - Math.PI / 6),

        y2 -
        headLength *
        Math.sin(angle - Math.PI / 6)
    );

    context.moveTo(
        x2,
        y2
    );

    context.lineTo(
        x2 -
        headLength *
        Math.cos(angle + Math.PI / 6),

        y2 -
        headLength *
        Math.sin(angle + Math.PI / 6)
    );

    context.stroke();
}


// Setup canvas
function setupShapeCanvas() {

    const image =
        document.getElementById("editorImage");

    const canvas =
        document.getElementById("shapeCanvas");

    canvas.width =
        image.clientWidth;

    canvas.height =
        image.clientHeight;

    canvas.style.width =
        image.clientWidth + "px";

    canvas.style.height =
        image.clientHeight + "px";
}


// Mouse released
document.addEventListener(
    "mouseup",
    function () {

        if (!isDrawingShape) {
            return;
        }

        isDrawingShape = false;

        document
            .getElementById("shapeActions")
            .style.display = "flex";

        console.log(
            "Shape drawing finished:",
            currentShape
        );
    }
);


// Cancel shape
function cancelShape() {

    const canvas =
        document.getElementById("shapeCanvas");

    const context =
        canvas.getContext("2d");

    // Clear preview
    context.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvas.style.display = "none";


    // Hide buttons
    document
        .getElementById("shapeActions")
        .style.display = "none";


    // Exit shape mode
    shapeMode = false;
    isDrawingShape = false;
    currentShape = null;
    deactivateTool("shape");

    const image =
        document.getElementById("editorImage");

    image.style.cursor = "default";

    image.removeEventListener(
        "mousedown",
        startShapeDrawing
    );


    showNotification(
        "Shape cancelled"
    );
}

async function applyShape() {

    if (!currentShape) {
        return;
    }


    const image =
        document.getElementById("editorImage");

    const canvas =
        document.getElementById("shapeCanvas");

    const rect =
        image.getBoundingClientRect();


    // Convert displayed coordinates
    // to original image coordinates
    const scaleX =
        image.naturalWidth / rect.width;

    const scaleY =
        image.naturalHeight / rect.height;


    let shapeData = {
        shape: currentShape,

        color: [0, 0, 255],

        thickness: Math.max(2, Math.round(2 * scaleX))
    };


    // -------------------------
    // Rectangle
    // -------------------------

    if (currentShape === "rectangle") {

        shapeData.x =
            Math.round(
                Math.min(startX, endX) * scaleX
            );

        shapeData.y =
            Math.round(
                Math.min(startY, endY) * scaleY
            );

        shapeData.width =
            Math.round(
                Math.abs(endX - startX) * scaleX
            );

        shapeData.height =
            Math.round(
                Math.abs(endY - startY) * scaleY
            );
    }


    // -------------------------
    // Circle
    // -------------------------

    else if (currentShape === "circle") {

        shapeData.center_x =
            Math.round(
                startX * scaleX
            );

        shapeData.center_y =
            Math.round(
                startY * scaleY
            );


        const radius =
            Math.sqrt(
                Math.pow(
                    endX - startX,
                    2
                ) +

                Math.pow(
                    endY - startY,
                    2
                )
            );


        shapeData.radius =
            Math.round(
                radius * scaleX
            );
    }


    // -------------------------
    // Arrow
    // -------------------------

    else if (currentShape === "arrow") {

        shapeData.x1 =
            Math.round(
                startX * scaleX
            );

        shapeData.y1 =
            Math.round(
                startY * scaleY
            );

        shapeData.x2 =
            Math.round(
                endX * scaleX
            );

        shapeData.y2 =
            Math.round(
                endY * scaleY
            );
    }


    console.log(
        "Sending shape:",
        shapeData
    );


    try {

        const response =
            await fetch(
                "/shape",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            shapeData
                        )
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            showNotification(
                data.message
            );

            return;
        }


        // Clear preview
        const context =
            canvas.getContext("2d");

        context.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );


        // Hide canvas
        canvas.style.display =
            "none";


        // Hide buttons
        document
            .getElementById("shapeActions")
            .style.display = "none";


        // Exit shape mode
        shapeMode = false;
        isDrawingShape = false;
        currentShape = null;

        deactivateTool("shape");

        image.style.cursor =
            "default";


        image.removeEventListener(
            "mousedown",
            startShapeDrawing
        );


        // Update image
        image.src =
            data.image_url +
            "?t=" +
            new Date().getTime();
        showNotification(
            data.message
        );

    } catch (error) {

        console.error(error);

        showNotification(
            "Failed to add shape."
        );
    }
}