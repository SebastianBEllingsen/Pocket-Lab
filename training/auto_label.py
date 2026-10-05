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
MAX_RBC_AREA = 12000


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

def find_rbc_seeds(area):
    """
    One seed per RBC centre. A peak in the distance map at distance d means a
    cell of radius ~d, so any weaker peak closer than that is the same cell.
    """
    dist = cv2.distanceTransform(area, cv2.DIST_L2, 5)
    local_max = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ys, xs = np.nonzero((dist == local_max) & (dist > 8))

    seeds = np.zeros_like(area)
    kept = []
    for i in np.argsort(-dist[ys, xs]):  # strongest peak first
        x, y, d = xs[i], ys[i], dist[ys[i], xs[i]]
        if all(np.hypot(x - kx, y - ky) > max(d, kd) for kx, ky, kd in kept):
            kept.append((x, y, d))
            cv2.circle(seeds, (int(x), int(y)), 4, 255, -1)
    return seeds


def remove_nested_rbcs(boxes):
    """
    Drop small RBC pieces: boxes under half the normal RBC size that lie
    mostly (70%+) inside a bigger RBC box are part of that same cell.
    """
    rbc = [b for b in boxes if b[0] == CLASSES.index("rbc")]
    if not rbc:
        return boxes
    area = lambda b: (b[3] - b[1]) * (b[4] - b[2])
    normal = np.median([area(b) for b in rbc])

    pieces = set()
    for small in rbc:
        if area(small) >= 0.5 * normal:
            continue
        for big in rbc:
            overlap = (max(0, min(small[3], big[3]) - max(small[1], big[1]))
                       * max(0, min(small[4], big[4]) - max(small[2], big[2])))
            if big is not small and area(big) > area(small) and overlap >= 0.7 * area(small):
                pieces.add(small)

    return [b for b in boxes if b not in pieces]


def flood(image, markers, cells):
    """Watershed: grow every seed in markers until the regions meet."""
    markers = markers.copy()
    markers[(cells > 0) & (markers == 1)] = 0  # let watershed decide these
    return cv2.watershed(image, markers.astype(np.int32))


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

    rbc_seeds = find_rbc_seeds(seed_area)

    _, markers = cv2.connectedComponents(rbc_seeds)
    markers = markers + 1  # 1 = background
    wbc_label = markers.max() + 1
    if wbc_marker is not None:
        markers[wbc_marker > 0] = wbc_label

    # 1. White cell: flood by image colour, which follows its edge well
    wbc_region = np.zeros_like(cells)
    if wbc_marker is not None:
        by_colour = flood(img, markers, cells)
        wbc_region = np.uint8((by_colour == wbc_label) & (cells > 0)) * 255 | wbc_marker
        x, y, w, h = cv2.boundingRect(wbc_region)
        if w * h > 0.3 * SIZE * SIZE:
            return img, None  # probably two touching white cells
        boxes.append((CLASSES.index(folder), x, y, x + w, y + h))

    # 2. RBCs: remove the white cell, then flood the distance map, which
    #    fills whole RBCs and ignores their pale centres
    rbc_area = cv2.bitwise_and(cells, cv2.bitwise_not(wbc_region))
    rbc_markers = markers.copy()
    rbc_markers[rbc_markers == wbc_label] = 1
    landscape = cv2.distanceTransform(rbc_area, cv2.DIST_L2, 5)
    landscape = np.uint8(255 - cv2.normalize(landscape, None, 0, 255, cv2.NORM_MINMAX))
    by_distance = flood(cv2.cvtColor(landscape, cv2.COLOR_GRAY2BGR), rbc_markers, rbc_area)

    for label in np.unique(by_distance):
        if label <= 1:
            continue
        region = np.uint8((by_distance == label) & (rbc_area > 0)) * 255
        x, y, w, h = cv2.boundingRect(region)
        if w > 0 and h > 0 and MIN_RBC_AREA < cv2.countNonZero(region) < MAX_RBC_AREA:
            boxes.append((CLASSES.index("rbc"), x, y, x + w, y + h))

    # Anything whose centre lies inside the white cell is part of it
    # (granules are not platelets, cytoplasm pieces are not RBCs)
    wbc = [b for b in boxes if b[0] >= 2]
    if wbc:
        _, wx1, wy1, wx2, wy2 = wbc[0]
        boxes = [b for b in boxes if b[0] >= 2 or not (
            wx1 <= (b[1] + b[3]) / 2 <= wx2 and wy1 <= (b[2] + b[4]) / 2 <= wy2)]

    boxes = remove_nested_rbcs(boxes)

    return img, boxes

