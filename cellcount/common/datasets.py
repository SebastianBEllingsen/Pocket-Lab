"""Dataset access behind one interface.

Every dataset is exposed as a list of ``Sample`` objects, so stage code never
touches paths or file layouts. New datasets are added by writing one loader
function and registering it with ``@register_dataset``.

Public API
    Box                       one annotated box (label + pixel corners)
    Sample                    one image plus optional reference mask / boxes
    resolve_data_root(arg)    --data-root flag > $CELLCOUNT_DATA_ROOT > <project>/Data_Sets
    available_datasets()      names accepted by get_dataset
    get_dataset(name, root)   list of Samples, sorted by name (stable indices)
    sample_from_path(path)    wrap an arbitrary image file as a Sample (no reference)

Dataset names (aliases in brackets)
    rbc-masks ("1")   Data_Set_1: purple smears with binary cell masks
    bccd ("2")        Data_Set_2/dataset-master: BCCD full fields with VOC boxes
    pbc ("3")         Data_Set_3: PBC single-cell images, label = class folder name
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Box:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class Sample:
    """One image and, when the dataset has them, its reference annotations."""

    dataset: str
    name: str
    image_path: Path
    mask_path: Path | None = None
    annotation_path: Path | None = None
    label: str | None = None   # class label for classification datasets (e.g. cell type)

    def load_image(self) -> np.ndarray:
        """BGR uint8 image."""
        img = cv.imread(str(self.image_path), cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {self.image_path}")
        return img

    def load_mask(self) -> np.ndarray | None:
        """Reference foreground mask as a bool array, or None if the dataset has none."""
        if self.mask_path is None:
            return None
        m = cv.imread(str(self.mask_path), cv.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Could not read mask: {self.mask_path}")
        return m > 127

    def load_boxes(self) -> list[Box]:
        """Annotated boxes (empty list if the dataset has none)."""
        if self.annotation_path is None:
            return []
        boxes = []
        for obj in ET.parse(self.annotation_path).getroot().iter("object"):
            bb = obj.find("bndbox")
            boxes.append(Box(
                label=(obj.findtext("name") or "?").strip(),
                xmin=int(float(bb.findtext("xmin"))), ymin=int(float(bb.findtext("ymin"))),
                xmax=int(float(bb.findtext("xmax"))), ymax=int(float(bb.findtext("ymax"))),
            ))
        return boxes


_LOADERS: dict[str, Callable[[Path], list[Sample]]] = {}
_ALIASES: dict[str, str] = {}


def register_dataset(name: str, *aliases: str):
    def deco(fn: Callable[[Path], list[Sample]]):
        _LOADERS[name] = fn
        _ALIASES[name] = name
        for a in aliases:
            _ALIASES[a] = name
        return fn
    return deco


def resolve_data_root(arg: str | os.PathLike | None = None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    env = os.environ.get("CELLCOUNT_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else PROJECT_ROOT / "Data_Sets"


def available_datasets() -> list[str]:
    return sorted(_ALIASES)


def get_dataset(name: str, data_root: Path | None = None) -> list[Sample]:
    key = _ALIASES.get(name)
    if key is None:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {', '.join(available_datasets())}")
    samples = _LOADERS[key](resolve_data_root(data_root))
    if not samples:
        raise FileNotFoundError(f"Dataset {key!r} found no images under {resolve_data_root(data_root)}")
    return sorted(samples, key=lambda s: s.name)


def sample_from_path(path: str | os.PathLike) -> Sample:
    p = Path(path).expanduser().resolve()
    return Sample(dataset="custom", name=p.stem, image_path=p)


@register_dataset("rbc-masks", "1")
def _load_rbc_masks(root: Path) -> list[Sample]:
    base = root / "Data_Set_1" / "Final data" / "Final data" / "New folder"
    masks = {p.stem: p for p in (base / "Mask").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    out = []
    # Some originals are .jpg while their mask is .png, so pair by file stem.
    for img in (base / "Original").glob("*"):
        if img.suffix.lower() in IMAGE_SUFFIXES and img.stem in masks:
            out.append(Sample("rbc-masks", img.stem, img, mask_path=masks[img.stem]))
    return out


@register_dataset("bccd", "2")
def _load_bccd(root: Path) -> list[Sample]:
    base = root / "Data_Set_2" / "dataset-master" / "dataset-master"
    out = []
    for img in (base / "JPEGImages").glob("*.jpg"):
        xml = base / "Annotations" / f"{img.stem}.xml"
        out.append(Sample("bccd", img.stem, img, annotation_path=xml if xml.exists() else None))
    return out


@register_dataset("pbc", "3")
def _load_pbc(root: Path) -> list[Sample]:
    base = root / "Data_Set_3" / "extracted" / "PBC_dataset_normal_DIB"
    return [Sample("pbc", img.stem, img, label=img.parent.name)
            for img in base.glob("*/*") if img.suffix.lower() in IMAGE_SUFFIXES]
