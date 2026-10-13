let activeTool = null;


function activateTool(tool) {

    // Check if another tool is already active
    if (activeTool !== null) {

        showNotification(
            "Finish or cancel the current tool first"
        );

        return false;
    }

    // Activate the tool
    activeTool = tool;

    return true;
}


function deactivateTool(tool) {

    // Only deactivate if this is the active tool
    if (activeTool === tool) {
        activeTool = null;
    }
}

