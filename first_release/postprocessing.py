"""
Post-processing that corrects YOLO's red cell boxes after detection.

YOLO's weak spot is clusters of overlapping or pale red cells: it puts one
box over 2-3 cells, or misses pale cells. This file builds a cell mask from
the image (steps 1-3; YOLO never sees this processed image) and uses it to
correct YOLO's boxes (step 4):

    1. enhance            -> cells bright, background black
    2. remove background  -> mask of all cells
    3. split the mask     -> one box per cell (distance map + watershed)
    4. refine YOLO        -> split YOLO boxes that hold 2+ cells,
                             add cells YOLO missed
"""
import cv2
import numpy as np

SIZE = 360  # processed at this size, so the pixel limits below fit
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MIN_SIZE = 0.6  # a mask cell counts only if its box is >= 60% of a typical red cell box


# ---------------------------------------------------------------------------
# 1 + 2. Enhance and remove the background
# ---------------------------------------------------------------------------

def enhance(img):
    """Grey image: brightness = how different a pixel's colour is from the background."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Background = typical colour of the brightest 25% of the image
    lightness = lab[:, :, 0]
    background = np.median(lab[lightness >= np.percentile(lightness, 75)], axis=0)

    distance = np.linalg.norm(lab - background, axis=2)
    distance = cv2.GaussianBlur(distance, (5, 5), 0)

    # Stretch x4 and cap at 255: even pale red cells become bright
    return np.uint8(np.clip(distance * 4, 0, 255))


def remove_background(enhanced):
    """Mask: white on cells, black on background."""
    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

    # Fill the pale red cell centres: any small background region is a hole
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 1500:
            mask[labels == i] = 255
    return mask


# ---------------------------------------------------------------------------
# 3. Split the mask into single cells
# ---------------------------------------------------------------------------

def split_cells(mask):
    """One box per cell in the mask: a seed in each cell's middle, grown with watershed."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # One seed per cell: a peak at distance d means a cell of radius ~d,
    # so a weaker peak closer than d belongs to the same cell
    local_max = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ys, xs = np.nonzero((dist == local_max) & (dist > 8))
    seeds = np.zeros_like(mask)
    kept = []
    for i in np.argsort(-dist[ys, xs]):
        x, y, d = xs[i], ys[i], dist[ys[i], xs[i]]
        if all(np.hypot(x - kx, y - ky) > max(d, kd) for kx, ky, kd in kept):
            kept.append((x, y, d))
            cv2.circle(seeds, (int(x), int(y)), 4, 255, -1)

    # Watershed on the distance map: fills whole cells, ignores pale centres
    _, markers = cv2.connectedComponents(seeds)
    markers = markers + 1                       # 1 = background
    markers[(mask > 0) & (markers == 1)] = 0
    landscape = np.uint8(255 - cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX))
    markers = cv2.watershed(cv2.cvtColor(landscape, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))

    boxes = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        region = np.uint8((markers == label) & (mask > 0)) * 255
        if cv2.countNonZero(region) > 0:
            x, y, w, h = cv2.boundingRect(region)
            boxes.append((x, y, x + w, y + h))
    return boxes


# ---------------------------------------------------------------------------
# 4. Refine YOLO's red cells
# ---------------------------------------------------------------------------

def centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def contains(box, point):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def remove_nested(boxes):
    """
    Drop boxes that lie mostly (70%+) inside a bigger box: when the watershed
    finds only part of a cell, that part sits inside the box of the whole
    cell, and counting both would count one cell twice.
    """
    nested = []
    for small in boxes:
        for big in boxes:
            overlap = (max(0, min(small[2], big[2]) - max(small[0], big[0]))
                       * max(0, min(small[3], big[3]) - max(small[1], big[1])))
            if area(big) > area(small) and overlap >= 0.7 * area(small):
                nested.append(small)
                break
    return [b for b in boxes if b not in nested]


def refine_red_cells(img, cells):
    """
    Take YOLO's cells and return a corrected list:
      - a YOLO red cell box holding 2+ full-size mask cells -> one box per cell
      - full-size mask cells that no YOLO box covers -> added (confidence 0)
    """
    yolo_rbc = [c for c in cells if c["type"] == "rbc"]
    others = [c for c in cells if c["type"] != "rbc"]
    if not yolo_rbc:
        return cells

    # Build the cell mask at 360 x 360
    small = cv2.resize(img, (SIZE, SIZE))
    sx, sy = img.shape[1] / SIZE, img.shape[0] / SIZE
    mask = remove_background(enhance(small))

    # White cells and platelets are YOLO's job: blank them out of the mask
    for c in others:
        x1, y1, x2, y2 = c["box"]
        mask[int(y1 / sy):int(y2 / sy), int(x1 / sx):int(x2 / sx)] = 0

    # Mask cells back in original coordinates, only the full-size ones
    typical = np.median([area(c["box"]) for c in yolo_rbc])
    mask_cells = [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                  for x1, y1, x2, y2 in split_cells(mask)]
    mask_cells = remove_nested([b for b in mask_cells if area(b) >= MIN_SIZE * typical])

    refined = []
    for y in yolo_rbc:
        inside = [b for b in mask_cells if contains(y["box"], centre(b))]
        if len(inside) >= 2:
            refined += [{"type": "rbc", "confidence": y["confidence"], "box": b} for b in inside]
        else:
            refined.append(y)

    covered = refined + others
    for b in mask_cells:
        if not any(contains(c["box"], centre(b)) for c in covered):
            refined.append({"type": "rbc", "confidence": 0.0, "box": b})

    return others + refined
