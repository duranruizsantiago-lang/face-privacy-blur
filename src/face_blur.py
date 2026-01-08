from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

import cv2
import numpy as np

from .utils import apply_face_blur, draw_bbox


# OpenCV pre-trained DNN face detector (ResNet SSD)
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"


def ensure_model_files(models_dir: Path) -> tuple[Path, Path]:
    """
    Ensures the DNN face detector files exist. Downloads them if missing.
    Returns (proto_path, model_path).
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    proto_path = models_dir / "deploy.prototxt"
    model_path = models_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    if not proto_path.exists():
        print("⬇️ Downloading deploy.prototxt ...")
        urllib.request.urlretrieve(PROTO_URL, proto_path)

    if not model_path.exists():
        print("⬇️ Downloading face detector model ... (this can take a moment)")
        urllib.request.urlretrieve(MODEL_URL, model_path)

    return proto_path, model_path


def load_face_net(proto_path: Path, model_path: Path):
    net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    return net


from typing import List, Tuple
import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def _nms(boxes: List[BBox], scores: List[float], iou_thresh: float = 0.35) -> List[int]:
    """Non-Max Suppression para quitar cajas duplicadas/encimadas."""
    if not boxes:
        return []

    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)

    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = s.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def detect_faces_bboxes_dnn(
    img_bgr,
    net,
    min_conf: float = 0.75,
    expand: float = 0.10,
    min_face_px: int = 40,
    aspect_min: float = 0.60,
    aspect_max: float = 1.80,
    verify_haar: bool = True,
) -> List[BBox]:
    """
    Detecta caras con DNN y (opcional) valida con Haar cascade para reducir falsos positivos.
    """
    h, w = img_bgr.shape[:2]

    # DNN forward
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    det = net.forward()

    boxes: List[BBox] = []
    scores: List[float] = []

    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < min_conf:
            continue

        x1 = int(det[0, 0, i, 3] * w)
        y1 = int(det[0, 0, i, 4] * h)
        x2 = int(det[0, 0, i, 5] * w)
        y2 = int(det[0, 0, i, 6] * h)

        # Clamp
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        bw, bh = (x2 - x1), (y2 - y1)
        if bw < min_face_px or bh < min_face_px:
            continue

        aspect = bw / max(1, bh)
        if not (aspect_min <= aspect <= aspect_max):
            continue

        # Expand bbox
        ex = int(bw * expand)
        ey = int(bh * expand)
        xx1 = max(0, x1 - ex)
        yy1 = max(0, y1 - ey)
        xx2 = min(w - 1, x2 + ex)
        yy2 = min(h - 1, y2 + ey)

        boxes.append((xx1, yy1, xx2, yy2))
        scores.append(conf)

    # NMS para quitar duplicados
    keep_idx = _nms(boxes, scores, iou_thresh=0.35)
    boxes = [boxes[k] for k in keep_idx]
    scores = [scores[k] for k in keep_idx]

    # Verificación Haar (reduce falsos positivos)
    if verify_haar and boxes:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        verified: List[BBox] = []
        for (x1, y1, x2, y2) in boxes:
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            found = cascade.detectMultiScale(
                roi,
                scaleFactor=1.1,
                minNeighbors=4,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(min_face_px, min_face_px),
            )
            if len(found) > 0:
                verified.append((x1, y1, x2, y2))

        boxes = verified

    return boxes


def process_image(
    input_path: Path,
    output_path: Path,
    min_conf: float,
    blur_k: int,
    draw: bool,
    expand: float,
    net,
) -> None:
    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    bboxes = detect_faces_bboxes_dnn(img, net, min_conf, expand)
    for bb in bboxes:
        apply_face_blur(img, bb, k=blur_k)
        if draw:
            draw_bbox(img, bb)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), img)
    if not ok:
        raise RuntimeError(f"Could not write output image: {output_path}")

    print(f"✅ Saved: {output_path} (faces blurred: {len(bboxes)})")


def process_video(
    source: str,
    output_path: Path | None,
    min_conf: float,
    blur_k: int,
    draw: bool,
    expand: float,
    net,
) -> None:
    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        bboxes = detect_faces_bboxes_dnn(frame, net, min_conf, expand)
        for bb in bboxes:
            apply_face_blur(frame, bb, k=blur_k)
            if draw:
                draw_bbox(frame, bb)

        if writer:
            writer.write(frame)

        cv2.imshow("Face Privacy Blur (press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"✅ Saved: {output_path}")
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Face Privacy Blur (OpenCV DNN + OpenCV)")
    parser.add_argument("--mode", choices=["image", "video", "webcam"], required=True)
    parser.add_argument("--input", help="Path to image/video. Not required for webcam.")
    parser.add_argument("--output", help="Output path. Optional for webcam.")
    parser.add_argument("--min-conf", type=float, default=0.6, help="Min detection confidence (0-1)")
    parser.add_argument("--blur-k", type=int, default=35, help="Gaussian blur kernel size (odd recommended)")
    parser.add_argument("--draw-box", action="store_true", help="Draw face bounding boxes")
    parser.add_argument("--expand", type=float, default=0.15, help="Expand bbox fraction (0.0-0.5)")

    args = parser.parse_args()

    proto_path, model_path = ensure_model_files(Path("models"))
    net = load_face_net(proto_path, model_path)

    if args.mode == "image":
        if not args.input:
            raise SystemExit("❌ --input is required for image mode")
        out = Path(args.output) if args.output else Path("outputs/blurred.jpg")
        process_image(Path(args.input), out, args.min_conf, args.blur_k, args.draw_box, args.expand, net)

    elif args.mode == "video":
        if not args.input:
            raise SystemExit("❌ --input is required for video mode")
        out = Path(args.output) if args.output else Path("outputs/blurred.mp4")
        process_video(args.input, out, args.min_conf, args.blur_k, args.draw_box, args.expand, net)

    else:  # webcam
        out = Path(args.output) if args.output else None
        process_video("webcam", out, args.min_conf, args.blur_k, args.draw_box, args.expand, net)


if __name__ == "__main__":
    main()
