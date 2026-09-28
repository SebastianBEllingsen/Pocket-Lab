import cv2
import numpy as np


def detect_cells(img):
    """Return (rbc_contours, wbc_contours) for a BGR image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # --- 1. White blood cells (purple-stained nucleus/granules) ---
    wbc_raw = cv2.inRange(hsv, (120, 60, 40), (165, 255, 255))
    wbc_raw = cv2.morphologyEx(wbc_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    wbc_raw = cv2.morphologyEx(wbc_raw, cv2.MORPH_CLOSE, kernel, iterations=3)
    wbc_raw = cv2.dilate(wbc_raw, kernel, iterations=2)

    wbc_candidates, _ = cv2.findContours(wbc_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wbc_contours = [cv2.convexHull(c) for c in wbc_candidates if cv2.contourArea(c) > 1500]

    wbc_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(wbc_mask, wbc_contours, -1, 255, cv2.FILLED)

    # --- 2. Segment all cells (green channel: cells darker than background) ---
    green = cv2.GaussianBlur(img[:, :, 1], (5, 5), 0)
    t, _ = cv2.threshold(green[wbc_raw == 0], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print("Otsu t:", t)
    thresh = np.uint8(green < t) * 255
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill pale RBC centers
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_mask = np.zeros_like(thresh)
    cv2.drawContours(cell_mask, contours, -1, 255, cv2.FILLED)

    # --- 3. Red blood cells = cells minus WBC area ---
    rbc_mask = cv2.bitwise_and(cell_mask, cv2.bitwise_not(wbc_mask))
    rbc_mask = cv2.morphologyEx(rbc_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- 4. Watershed with one seed per cell center (local maxima of distance map) ---
    dist = cv2.distanceTransform(rbc_mask, cv2.DIST_L2, 5)
    peaks = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    sure_fg = np.uint8((dist == peaks) & (dist > 8)) * 255
    sure_fg = cv2.dilate(sure_fg, kernel, iterations=2)

    sure_bg = cv2.dilate(rbc_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    min_area, max_area = 100, 12000
    rbc_contours = []
    for label in np.unique(markers):
        if label <= 1:  # -1 = boundaries, 1 = background
            continue
        region = np.uint8((markers == label) & (rbc_mask > 0)) * 255
        cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            rbc_contours.append(c)
        else:
            print("rejected area:", area)

    return rbc_contours, wbc_contours


def draw_cells(img, rbc_contours, wbc_contours, show_red=True, show_white=True):
    """Return a copy of img with the chosen contours drawn on it."""
    output = img.copy()
    # --- 5. Draw ---
    if show_red:
        cv2.drawContours(output, rbc_contours, -1, (0, 0, 255), 1, lineType=cv2.LINE_AA)  # red
    if show_white:
        cv2.drawContours(output, wbc_contours, -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)  # green

    return output
