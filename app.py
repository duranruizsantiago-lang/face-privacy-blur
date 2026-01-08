import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


# --- Model files (OpenCV DNN face detector) ---
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_samples_face_detector/res10_300x300_ssd_iter_140000.caffemodel"

def ensure_model_files(models_dir: Path) -> tuple[Path, Path]:
    # OFFLINE: no descarga nada, solo usa lo que ya existe en /models
    models_dir.mkdir(parents=True, exist_ok=True)

    proto_path = models_dir / "deploy.prototxt"
    model_path = models_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    if not proto_path.exists():
        raise FileNotFoundError(f"Falta el archivo: {proto_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Falta el archivo: {model_path}")

    return proto_path, model_path




@st.cache_resource
def load_net(proto_path: str, model_path: str):
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    return net


def detect_faces_bboxes_dnn(
    image_bgr: np.ndarray,
    net,
    min_conf: float = 0.6,
    nms_thresh: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """
    Returns list of bounding boxes as (x1, y1, x2, y2).
    Uses confidence threshold + NMS to reduce false positives/duplicates.
    """
    (h, w) = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    confs = []

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < min_conf:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype(int)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        boxes.append([x1, y1, bw, bh])  # NMSBoxes wants [x,y,w,h]
        confs.append(conf)

    if not boxes:
        return []

    idxs = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=min_conf, nms_threshold=nms_thresh)
    if len(idxs) == 0:
        return []

    out = []
    for idx in idxs.flatten():
        x, y, bw, bh = boxes[idx]
        out.append((x, y, x + bw, y + bh))

    return out


def blur_faces(
    image_bgr: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    blur_k: int = 35,
    expand: float = 0.15,
    draw_box: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Blurs face areas; expand grows the box by % to cover hair/edges.
    blur_k must be odd.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]

    # ensure odd blur kernel
    blur_k = int(blur_k)
    if blur_k < 3:
        blur_k = 3
    if blur_k % 2 == 0:
        blur_k += 1

    faces_blurred = 0

    for (x1, y1, x2, y2) in bboxes:
        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * expand)
        pad_y = int(bh * expand)

        xx1 = max(0, x1 - pad_x)
        yy1 = max(0, y1 - pad_y)
        xx2 = min(w, x2 + pad_x)
        yy2 = min(h, y2 + pad_y)

        roi = out[yy1:yy2, xx1:xx2]
        if roi.size == 0:
            continue

        blurred = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
        out[yy1:yy2, xx1:xx2] = blurred
        faces_blurred += 1

        if draw_box:
            cv2.rectangle(out, (xx1, yy1), (xx2, yy2), (0, 255, 0), 2)

    return out, faces_blurred


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def encode_jpg(img_bgr: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Could not encode image as JPG")
    return buf.tobytes()


def main():
    st.set_page_config(page_title="Face Privacy Blur", page_icon="🕶️", layout="centered")
    st.title("🕶️ Face Privacy Blur")
    st.write("Sube una imagen y te devuelve la misma imagen con las caras difuminadas (blur).")

    with st.sidebar:
        st.header("⚙️ Ajustes")
        min_conf = st.slider("Confianza mínima (menos falsos positivos)", 0.2, 0.95, 0.65, 0.05)
        nms_thresh = st.slider("NMS (quita duplicados)", 0.1, 0.7, 0.3, 0.05)
        blur_k = st.slider("Fuerza del blur (kernel impar)", 3, 99, 35, 2)
        expand = st.slider("Expandir caja (%)", 0.0, 0.6, 0.15, 0.05)
        draw_box = st.checkbox("Dibujar cajas (debug)", value=False)

        st.caption("Tip: si detecta caras donde no hay, sube la **confianza** a 0.75–0.85.")

    uploaded = st.file_uploader("📤 Sube tu imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.stop()

    img_bgr = decode_uploaded_image(uploaded.read())
    if img_bgr is None:
        st.error("No pude leer esa imagen. Prueba con otra (JPG/PNG estándar).")
        st.stop()

    st.subheader("Vista previa")
    st.image(bgr_to_rgb(img_bgr), caption="Original", use_container_width=True)

    if st.button("🔒 Aplicar blur"):
        models_dir = Path("models")
        proto_path, model_path = ensure_model_files(models_dir)
        net = load_net(str(proto_path), str(model_path))

        bboxes = detect_faces_bboxes_dnn(img_bgr, net, min_conf=min_conf, nms_thresh=nms_thresh)
        out_bgr, n = blur_faces(img_bgr, bboxes, blur_k=blur_k, expand=expand, draw_box=draw_box)

        st.success(f"✅ Listo. Caras difuminadas: {n}")
        st.image(bgr_to_rgb(out_bgr), caption="Resultado", use_container_width=True)

        out_bytes = encode_jpg(out_bgr, quality=95)
        st.download_button(
            "⬇️ Descargar imagen blur (JPG)",
            data=out_bytes,
            file_name="blurred.jpg",
            mime="image/jpeg",
        )


if __name__ == "__main__":
    main()
