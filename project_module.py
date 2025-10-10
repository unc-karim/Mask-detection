# Auto-generated from project.ipynb via nbconvert.

# === run_webcam entrypoint ===

import sys
from pathlib import Path
from typing import Tuple, Union

import cv2
import joblib
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torchvision import models

ModelPath = Union[Path, str]
SplitsPath = Union[Path, str]


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
    """Load a ResNet50 backbone without a classifier head."""
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
        raise ValueError('Empty face crop encountered.')

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor


def annotate_frame(
    frame: np.ndarray, bbox: Tuple[int, int, int, int], label: str, confidence: float
) -> None:
    """Draw bounding box and prediction label on the frame in-place."""
    x, y, w, h = bbox
    color = (0, 200, 0) if label == 'Mask' else (0, 0, 255)
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
    """Load classifier if present; otherwise train and persist it."""
    if model_path.exists():
        print(f"[info] Loading classifier: {model_path}")
        return joblib.load(model_path)

    expected = {
        'train': splits_dir / 'DS3_train.npz',
        'val': splits_dir / 'DS3_val.npz',
    }
    missing = [name for name, path in expected.items() if not path.exists()]
    if missing:
        hint = ', '.join(missing)
        raise FileNotFoundError(
            f"Missing required feature files in {splits_dir}: {hint}. "
            "Generate them by running feature extraction or provide a trained classifier."
        )

    def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path, allow_pickle=False)
        return data['X'], data['y']

    X_train, y_train = load_npz(expected['train'])
    X_val, y_val = load_npz(expected['val'])
    X_pool = np.concatenate([X_train, X_val], axis=0)
    y_pool = np.concatenate([y_train, y_val], axis=0)

    print(f"[info] Training LinearSVC on pooled features ({X_pool.shape[0]} samples).")
    pipeline = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('svm', LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=10000)),
        ]
    )
    pipeline.fit(X_pool, y_pool)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[info] Saved classifier to {model_path}")
    return pipeline


def run_webcam(
    model_path: ModelPath,
    splits_dir: SplitsPath,
    camera_index: int = 0,
    use_gpu: bool = True,
) -> None:
    """Run real-time mask detection using the webcam feed."""
    model_path = Path(model_path)
    splits_dir = Path(splits_dir)

    clf = load_or_train_classifier(model_path, splits_dir)

    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    feature_extractor = load_feature_extractor(device)
    transform = build_transform()

    cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError(f'Failed to load Haar cascade from {cascade_path}')

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera index {camera_index}')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Failed to read frame from camera. Exiting.')
                break

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
                    confidence = float(abs(decision[0])) if decision.ndim == 1 else float(max(decision[0]))
                    label = 'Mask' if int(pred) == 1 else 'No Mask'
                    annotate_frame(frame, (x, y, w, h), label, confidence)
                except Exception as err:  # keep the loop running
                    print(f"[warn] Skipping face: {err}", file=sys.stderr)
                    continue

            cv2.imshow('Mask Detector', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

