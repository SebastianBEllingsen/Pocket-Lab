"""
Automatically draw a box around every cell in one PBC dataset image.

Uses what the dataset already tells us: each image (except platelet/)
has exactly one white cell in the centre, and the folder name is its type.
"""
import cv2
import numpy as np

SIZE = 360  # all images are resized to this, so the thresholds below fit

CLASSES = [
    "rbc", "platelet",
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil",
]

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

MIN_NUCLEUS_AREA = 1200   # purple group bigger than this = a WBC nucleus
MAX_PLATELET_AREA = 600   # purple blob smaller than this = a platelet
MIN_RBC_AREA = 600        # smaller "cells" are fragments, not RBCs
MAX_RBC_AREA = 6000


def purple_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (120, 60, 40), (165, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)


def cell_mask(img):
    """All cells = pixels whose colour is far from the background colour."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Background = typical colour of the brightest 25% of the image
    lightness = lab[:, :, 0]
    background = np.median(lab[lightness >= np.percentile(lightness, 75)], axis=0)

    distance = np.linalg.norm(lab - background, axis=2)
    distance = cv2.GaussianBlur(distance, (5, 5), 0)
    distance = np.uint8(np.clip(distance * 4, 0, 255))
    _, mask = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

    # Fill the pale RBC centres: any small background region is a hole
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 1500:
            mask[labels == i] = 255
    return mask


def find_nuclei(purple):
    """Group nucleus lobes together and return the big purple groups."""
    grouped = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, KERNEL, iterations=4)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(grouped)
    nuclei = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > MIN_NUCLEUS_AREA:
            nuclei.append((np.uint8(labels == i) * 255, centroids[i]))
    return nuclei


def find_platelets(purple, exclude):
    """Small purple blobs that are not part of a nucleus."""
    free = cv2.bitwise_and(purple, cv2.bitwise_not(exclude))
    n, _, stats, _ = cv2.connectedComponentsWithStats(free)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 40 < area < MAX_PLATELET_AREA:
            boxes.append((x, y, x + w, y + h))
    return boxes


def grow_wbc(nucleus, cells):
    """
    The nucleus is only part of the white cell. Estimate the whole cell as a
    circle around the nucleus: its radius is how far the widest point of the
    nucleus is from the nearest background pixel.
    """
    dist = cv2.distanceTransform(cells, cv2.DIST_L2, 5)
    dist_in_nucleus = np.where(nucleus > 0, dist, 0)
    _, radius, _, (cx, cy) = cv2.minMaxLoc(dist_in_nucleus)

    marker = nucleus.copy()                          # watershed seed
    cv2.circle(marker, (cx, cy), int(radius * 0.85), 255, -1)

    body = cv2.dilate(nucleus, KERNEL, iterations=3)  # no RBC seeds in here
    cv2.circle(body, (cx, cy), int(radius * 1.1), 255, -1)
    return marker, body


def label_image(img, folder):
    """
    Return (img_resized, [(class_id, x1, y1, x2, y2), ...])
    or (img_resized, None) if the image is too unclear to label safely.
    """
    img = cv2.resize(img, (SIZE, SIZE))
    purple = purple_mask(img)
    cells = cell_mask(img)
    centre = np.array([SIZE / 2, SIZE / 2])
    boxes = []

    nuclei = find_nuclei(purple)

    if folder == "platelet":
        # No WBC here: every purple blob, also clumps, is a platelet
        if any(cv2.countNonZero(m) > 4000 for m, _ in nuclei):
            return img, None  # something big and purple -> probably a WBC
        n, _, stats, _ = cv2.connectedComponentsWithStats(purple)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area > 15:
                boxes.append((CLASSES.index("platelet"), x, y, x + w, y + h))
        wbc_marker = None
    else:
        # Exactly one nucleus expected, close to the centre
        if len(nuclei) != 1:
            return img, None
        mask, c = nuclei[0]
        if np.linalg.norm(c - centre) > SIZE * 0.25:
            return img, None
        wbc_marker, wbc_body = grow_wbc(mask, cells)
        for x1, y1, x2, y2 in find_platelets(purple, wbc_body):
            boxes.append((CLASSES.index("platelet"), x1, y1, x2, y2))

    # --- Watershed: one seed per RBC centre + one seed for the WBC nucleus ---
    seed_area = cells.copy()
    if wbc_marker is not None:
        seed_area[wbc_body > 0] = 0

    dist = cv2.distanceTransform(seed_area, cv2.DIST_L2, 5)
    peaks = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    rbc_seeds = np.uint8((dist == peaks) & (dist > 8)) * 255
    rbc_seeds = cv2.dilate(rbc_seeds, KERNEL, iterations=2)

    _, markers = cv2.connectedComponents(rbc_seeds)
    markers = markers + 1                      # 1 = background
    wbc_label = markers.max() + 1
    if wbc_marker is not None:
        markers[wbc_marker > 0] = wbc_label
    unknown = cv2.dilate(cells, KERNEL, iterations=2) > 0
    markers[unknown & (markers == 1)] = 0      # let watershed decide these
    markers = cv2.watershed(img, markers.astype(np.int32))

    for label in np.unique(markers):
        if label <= 1:
            continue
        region = np.uint8((markers == label) & (cells > 0)) * 255
        if label == wbc_label:
            region |= wbc_marker
        x, y, w, h = cv2.boundingRect(region)
        if w == 0 or h == 0:
            continue
        area = cv2.countNonZero(region)

        if label == wbc_label:
            if w * h > 0.3 * SIZE * SIZE:
                return img, None  # probably two touching white cells
            boxes.append((CLASSES.index(folder), x, y, x + w, y + h))
        elif MIN_RBC_AREA < area < MAX_RBC_AREA:
            boxes.append((CLASSES.index("rbc"), x, y, x + w, y + h))

    # Anything whose centre lies inside the white cell is part of it
    # (granules are not platelets, cytoplasm pieces are not RBCs)
    wbc = [b for b in boxes if b[0] >= 2]
    if wbc:
        _, wx1, wy1, wx2, wy2 = wbc[0]
        boxes = [b for b in boxes if b[0] >= 2 or not (
            wx1 <= (b[1] + b[3]) / 2 <= wx2 and wy1 <= (b[2] + b[4]) / 2 <= wy2)]

    return img, boxes

