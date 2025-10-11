from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "OpenCV (cv2) is required. Install it via 'pip install opencv-python'."
    ) from exc

try:
    import joblib
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("joblib is required. Install it via 'pip install joblib'.") from exc

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "scikit-learn is required. Install it via 'pip install scikit-learn'."
    ) from exc


def build_transform() -> T.Compose:
    """Return the evaluation transform used during feature extraction."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def load_feature_extractor(device: torch.device) -> torch.nn.Module:
    """
    Load a ResNet50 backbone with the classification head removed so we can
    reuse it as a feature extractor.
    """
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    resnet.to(device)
    return resnet


def prepare_face(
    frame: np.ndarray, bbox: Tuple[int, int, int, int], transform: T.Compose
) -> torch.Tensor:
    """Crop, convert and transform a detected face ROI."""
    x, y, w, h = bbox
    face = frame[y : y + h, x : x + w]
    if face.size == 0:
        raise ValueError("Empty face crop encountered.")

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor


def annotate_frame(
    frame: np.ndarray, bbox: Tuple[int, int, int, int], label: str, confidence: float
) -> None:
    """Draw bounding box and prediction label on the frame in-place."""
    x, y, w, h = bbox
    color = (0, 200, 0) if label == "Mask" else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(
        frame,
        text,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def load_or_train_classifier(model_path: Path, splits_dir: Path) -> Pipeline:
    """
    Load a serialized classifier if available, otherwise train one from the
    saved feature splits and persist it to model_path.
    """
    if model_path.exists():
        print(f"[info] Loading classifier: {model_path}")
        return joblib.load(model_path)

    expected_files = {
        "train": splits_dir / "DS3_train.npz",
        "val": splits_dir / "DS3_val.npz",
    }
    missing = [name for name, p in expected_files.items() if not p.exists()]
    if missing:
        hint = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required feature files in {splits_dir}: {hint}. "
            "Generate them by running the feature extraction notebook or pass "
            "an explicit --model-path to a trained classifier."
        )

    def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path, allow_pickle=False)
        return data["X"], data["y"]

    X_train, y_train = load_npz(expected_files["train"])
    X_val, y_val = load_npz(expected_files["val"])
    X_pool = np.concatenate([X_train, X_val], axis=0)
    y_pool = np.concatenate([y_train, y_val], axis=0)

    print(f"[info] Training LinearSVC on pooled features ({X_pool.shape[0]} samples).")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(dual=False, C=1.0, class_weight="balanced", max_iter=10000)),
        ]
    )
    pipeline.fit(X_pool, y_pool)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[info] Saved classifier to {model_path}")
    return pipeline


def run_webcam(model_path: Path, splits_dir: Path, camera_index: int, use_gpu: bool) -> None:
    clf = load_or_train_classifier(model_path, splits_dir)

    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    feature_extractor = load_feature_extractor(device)
    transform = build_transform()

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera. Exiting.")
                break

            # Correct webcam mirror effect so the on-screen preview matches real-world orientation.
            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )

            for (x, y, w, h) in faces:
                try:
                    face_tensor = prepare_face(frame, (x, y, w, h), transform).to(device)
                    with torch.no_grad():
                        feat = feature_extractor(face_tensor).cpu().numpy()
                    pred = clf.predict(feat)[0]
                    decision = clf.decision_function(feat)
                    if decision.ndim == 1:
                        confidence = float(abs(decision[0]))
                    else:
                        confidence = float(max(decision[0]))
                    label = "Mask" if int(pred) == 1 else "No Mask"
                    annotate_frame(frame, (x, y, w, h), label, confidence)
                except Exception as err:
                    # Skip problematic detections but keep the loop running.
                    print(f"[warn] Skipping face: {err}", file=sys.stderr)
                    continue

            cv2.imshow("Mask Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time mask detection on the webcam feed.")
    default_model = Path(__file__).with_name("models") / "mask_svm.joblib"
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model,
        help=f"Path to the saved LinearSVC pipeline (.joblib). (default: {default_model})",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    default_splits = Path(__file__).resolve().parent / "data" / "splits"
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=default_splits,
        help=f"Directory containing DS3_train.npz/DS3_val.npz (default: {default_splits}).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_webcam(args.model_path, args.splits_dir, args.camera_index, use_gpu=not args.cpu)


if __name__ == "__main__":
    main()
