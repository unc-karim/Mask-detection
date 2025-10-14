from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models

try:
    import cv2
except ImportError as exc:
    raise SystemExit("OpenCV (cv2) is required. Install it via 'pip install opencv-python'.") from exc

try:
    import joblib
except ImportError as exc:
    raise SystemExit("joblib is required. Install it via 'pip install joblib'.") from exc


# ---------------------------
# Pure inference transforms
# ---------------------------
def build_transform(mean=None, std=None, size: int = 224):
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    return T.Compose([T.Resize(256), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])


# ---------------------------
# Load fixed backbone (no training)
# ---------------------------
def load_feature_extractor(device: torch.device, backbone_path: Optional[Path] = None):
    """
    Load a ResNet50 feature extractor strictly for inference.
    If `backbone_path` is provided, load state_dict from that file (exported in the notebook).
    Otherwise, use torchvision's fixed ImageNet weights (no training).
    """
    resnet = models.resnet50(weights=None if backbone_path else models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = torch.nn.Identity()
    resnet.eval().to(device)

    if backbone_path:
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone weights not found at {backbone_path}")
        state = torch.load(backbone_path, map_location=device)
        # Support either a plain state_dict or {'state_dict': ...}
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        resnet.load_state_dict(state_dict, strict=True)

    for p in resnet.parameters():
        p.requires_grad_(False)
    return resnet


# ---------------------------
# Load fixed classifier (no training)
# ---------------------------
def load_classifier(model_path: Path):
    """
    Load a trained classifier/pipeline that was exported from a separate notebook.
    Accepts either:
      - a scikit-learn Pipeline, or
      - a dict containing {'pipeline', 'classes', 'transform', 'feature_dim', ...}
    """
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise SystemExit("scikit-learn is required to load the trained pipeline. Install 'scikit-learn'.") from exc

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Export your trained model from the notebook "
            "(e.g., joblib.dump({'pipeline': pipe, ...}, 'models/mask_svm.joblib'))."
        )
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj
    # Wrap raw pipeline with sensible defaults
    return {"pipeline": obj, "classes": {0: "No Mask", 1: "Mask"}, "transform": {"mean": [0.485,0.456,0.406], "std":[0.229,0.224,0.225], "size":224}}


# ---------------------------
# Pre/Post utils
# ---------------------------
def prepare_face(frame: np.ndarray, bbox: Tuple[int, int, int, int], transform: T.Compose):
    x, y, w, h = bbox
    face = frame[y : y + h, x : x + w]
    if face.size == 0:
        raise ValueError("Empty face crop encountered.")
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    return transform(pil_img).unsqueeze(0)

def annotate_frame(frame: np.ndarray, bbox: Tuple[int, int, int, int], label: str, confidence: float):
    x, y, w, h = bbox
    color = (0, 200, 0) if label.lower().startswith("mask") else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ---------------------------
# Main loop (inference only)
# ---------------------------
def run_webcam(model_path: Path, camera_index: int, use_gpu: bool, backbone_path: Optional[Path]):
    export = load_classifier(model_path)
    clf = export["pipeline"]
    classes = export.get("classes", {0: "No Mask", 1: "Mask"})
    tmeta = export.get("transform", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "size": 224})

    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    feature_extractor = load_feature_extractor(device, backbone_path=backbone_path)
    transform = build_transform(tmeta.get("mean"), tmeta.get("std"), tmeta.get("size", 224))

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera. Exiting.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                try:
                    face_tensor = prepare_face(frame, (x, y, w, h), transform).to(device)
                    with torch.no_grad():
                        feat = feature_extractor(face_tensor).cpu().numpy()  # shape (1, 2048)

                    pred = clf.predict(feat)[0]

                    # Prefer calibrated probabilities; fallback to decision margins; else 0.5
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(feat)[0]
                        conf = float(np.max(proba))
                    elif hasattr(clf, "decision_function"):
                        dec = clf.decision_function(feat)
                        conf = float(abs(dec[0])) if dec.ndim == 1 else float(np.max(dec[0]))
                    else:
                        conf = 0.5

                    label = classes.get(int(pred), str(int(pred)))
                    annotate_frame(frame, (x, y, w, h), label, conf)

                except Exception as err:
                    print(f"[warn] Skipping face: {err}", file=sys.stderr)
                    continue

            cv2.imshow("Mask Detector (inference-only)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Run real-time mask detection using pre-trained artifacts only.")
    default_model = Path(__file__).with_name("models") / "mask_svm.joblib"
    parser.add_argument("--model-path", type=Path, default=default_model,
                        help=f"Path to exported classifier (.joblib). Default: {default_model}")
    parser.add_argument("--backbone-path", type=Path, default=None,
                        help="Optional path to exported ResNet50 state_dict (.pt) from the notebook. If omitted, "
                             "uses torchvision ImageNet weights (still inference-only).")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (default: 0).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_webcam(args.model_path, args.camera_index, use_gpu=not args.cpu, backbone_path=args.backbone_path)


if __name__ == "__main__":
    main()
