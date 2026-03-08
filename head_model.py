"""
FacePlay — head direction ML predictor
═══════════════════════════════════════
Wraps the scikit-learn model trained by collect_and_train.py.
Loaded once at startup by main.py via head_model.load().

Public API
──────────
    head_model.load()                   → loads head_model.pkl from disk
    head_model.predict(face_landmarks)  → (direction_str, approx_yaw_deg)
"""

import pickle
import os
import numpy as np

# ── File path ────────────────────────────────────────────────────────
MODEL_FILE = 'head_model.pkl'

# ── Private state ────────────────────────────────────────────────────
_model   = None
_scaler  = None
_lm_ids  = None   # landmark indices saved at training time


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def load(path: str = MODEL_FILE) -> None:
    """
    Load the trained model from disk.
    Raises FileNotFoundError if head_model.pkl is missing — run
    collect_and_train.py first.
    """
    global _model, _scaler, _lm_ids

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nhead_model.pkl not found at '{path}'.\n"
            "Run collect_and_train.py first to train the model."
        )

    with open(path, 'rb') as f:
        data = pickle.load(f)

    _model  = data['model']
    _scaler = data['scaler']
    _lm_ids = data['landmark_ids']
    print(f"head_model: loaded '{path}'  "
          f"({len(_lm_ids)} landmarks → {len(_lm_ids)*2} features, "
          f"{len(_model.classes_)} classes)")


def predict(face_landmarks) -> tuple[str, float, float]:
    """
    Predict head direction from face landmarks.

    Parameters
    ──────────
    face_landmarks : accepts either format —
        • FaceLandmarker (task API): fl_results.face_landmarks[0]  → list
        • FaceMesh (solutions API):  mesh_results.multi_face_landmarks[0] → proto

    Returns
    ───────
    direction   : str   — one of FORWARD / LEFT / RIGHT / UP / DOWN /
                                 UP_LEFT / UP_RIGHT
    approx_yaw  : float — rough yaw in pseudo-degrees for brow threshold.
                          Positive = looking right, negative = looking left.
    confidence  : float — model's probability for the winning class [0..1].
                          main.py gates direction changes on this value so
                          ambiguous frames between directions are ignored.
    """
    if _model is None:
        raise RuntimeError(
            "head_model not loaded. Call head_model.load() before predict()."
        )

    features = _extract_features(face_landmarks)
    if features is None:
        return "FORWARD", 0.0, 1.0

    scaled     = _scaler.transform(features.reshape(1, -1))
    probs      = _model.predict_proba(scaled)[0]
    confidence = float(probs.max())
    direction  = _model.classes_[probs.argmax()]

    # Approximate yaw from nose-tip offset relative to ear midpoint.
    lms    = face_landmarks.landmark if hasattr(face_landmarks, 'landmark') else face_landmarks
    l_ear  = lms[234]; r_ear = lms[454]
    cx     = (l_ear.x + r_ear.x) / 2.0
    fw     = abs(r_ear.x - l_ear.x)
    nose_x = lms[4].x
    approx_yaw = ((nose_x - cx) / fw) * 90.0 if fw > 1e-6 else 0.0

    return direction, approx_yaw, confidence


# ═══════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════

def _extract_features(face_landmarks) -> "np.ndarray | None":
    """
    40-element float32 feature vector: (x, y) for each of the 20 key
    landmarks, translated to face-centre and scaled by face width.

    OPTIMISED: uses 4 fixed anchor landmarks (ears 234/454, forehead 10,
    chin 152) for O(4) normalization instead of iterating all 478 landmarks.
    Must match collect_and_train.py exactly (same anchor formula).

    Accepts both FaceLandmarker list format and FaceMesh proto format.
    """
    lms = face_landmarks.landmark if hasattr(face_landmarks, 'landmark') else face_landmarks

    # O(4) anchor-based normalization — left ear, right ear, forehead, chin
    l_ear = lms[234]; r_ear = lms[454]
    cx    = (l_ear.x + r_ear.x) / 2.0
    cy    = (lms[10].y  + lms[152].y) / 2.0
    scale = abs(r_ear.x - l_ear.x)
    if scale < 1e-6:
        return None

    features = []
    for idx in _lm_ids:
        lm = lms[idx]
        features.append((lm.x - cx) / scale)
        features.append((lm.y - cy) / scale)
    return np.array(features, dtype=np.float32)
