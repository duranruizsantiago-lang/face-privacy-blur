from __future__ import annotations

import cv2
import numpy as np


def apply_face_blur(img: np.ndarray, bbox: tuple[int, int, int, int], k: int = 35) -> None:
    """
    Blur (in-place) a face region inside an image.
    bbox: (x1, y1, x2, y2) in pixel coordinates
    k: Gaussian blur kernel size (odd number recommended)
    """
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2]

    if k % 2 == 0:
        k += 1
    k = max(3, k)

    roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
    img[y1:y2, x1:x2] = roi_blur


def draw_bbox(img: np.ndarray, bbox: tuple[int, int, int, int], thickness: int = 2) -> None:
    """Draw a bounding box on the image (in-place)."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
