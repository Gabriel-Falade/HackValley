"""
FacePlay — head direction data collector + model trainer
═════════════════════════════════════════════════════════
Run this ONCE before launching main.py (or any time you want to retrain).

TWO ROUNDS PER DIRECTION
  Round 1 — hold the position steadily (baseline)
  Round 2 — drift and vary within the direction (edge cases)
  Both rounds are needed. The model must learn both the "clean" centre
  of each direction AND the messy edges near the boundaries.

DATA AUGMENTATION
  Each real frame is jittered 5× with small Gaussian noise before
  training.  This triples effective data without extra recording time
  and makes the model robust to minor camera-distance changes.

Usage:
    python collect_and_train.py

Requirements (in addition to the FacePlay dependencies):
    pip install scikit-learn
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

COLLECT_SECONDS   = 5    # seconds to record per direction per round
COUNTDOWN_SECONDS = 3    # "get ready" pause before each recording
MODEL_FILE        = 'head_model.pkl'

# Number of training rounds through all directions.
# Round 1 = hold still.  Round 2 = drift/vary.  Round 3 = wider drift.
ROUNDS = 3

# Data augmentation: copies per real frame.
# Each copy adds small Gaussian noise (σ = AUG_NOISE_STD).
AUG_COPIES    = 8
AUG_NOISE_STD = 0.003   # ~0.3 % of face width — tighter noise for sharper boundaries

# All 9 directions FacePlay needs to classify
DIRECTIONS = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "UP_LEFT",
    "UP_RIGHT",
    "DOWN_LEFT",
    "DOWN_RIGHT",
]

# Round 1: settle into the exact position
ROUND1_PROMPTS = {
    "FORWARD":    "Look straight at the camera — hold still",
    "LEFT":       "Turn head LEFT — hold still",
    "RIGHT":      "Turn head RIGHT — hold still",
    "UP":         "Tilt head UP — hold still",
    "DOWN":       "Tilt head DOWN — hold still",
    "UP_LEFT":    "Look UP and LEFT diagonally — hold still",
    "UP_RIGHT":   "Look UP and RIGHT diagonally — hold still",
    "DOWN_LEFT":  "Look DOWN and LEFT diagonally — hold still",
    "DOWN_RIGHT": "Look DOWN and RIGHT diagonally — hold still",
}

# Round 2: vary naturally within the direction
ROUND2_PROMPTS = {
    "FORWARD":    "Look forward — drift side to side slightly",
    "LEFT":       "Keep looking LEFT — vary the angle a little",
    "RIGHT":      "Keep looking RIGHT — vary the angle a little",
    "UP":         "Keep looking UP — nod slightly",
    "DOWN":       "Keep looking DOWN — nod slightly",
    "UP_LEFT":    "Keep looking UP-LEFT — vary the diagonal slightly",
    "UP_RIGHT":   "Keep looking UP-RIGHT — vary the diagonal slightly",
    "DOWN_LEFT":  "Keep looking DOWN-LEFT — vary the diagonal slightly",
    "DOWN_RIGHT": "Keep looking DOWN-RIGHT — vary the diagonal slightly",
}

# Round 3: push toward the edges of each direction (near-boundary data)
# This teaches the model where each class ends, sharpening decision boundaries.
ROUND3_PROMPTS = {
    "FORWARD":    "Look forward — hold as neutral/still as possible",
    "LEFT":       "Look LEFT — try a more extreme turn than before",
    "RIGHT":      "Look RIGHT — try a more extreme turn than before",
    "UP":         "Look UP — try tilting higher than before",
    "DOWN":       "Look DOWN — try tilting lower than before",
    "UP_LEFT":    "Look UP-LEFT — vary between more UP and more LEFT",
    "UP_RIGHT":   "Look UP-RIGHT — vary between more UP and more RIGHT",
    "DOWN_LEFT":  "Look DOWN-LEFT — vary between more DOWN and more LEFT",
    "DOWN_RIGHT": "Look DOWN-RIGHT — vary between more DOWN and more RIGHT",
}

ROUND_PROMPTS = [ROUND1_PROMPTS, ROUND2_PROMPTS, ROUND3_PROMPTS]

# 20 key FaceMesh landmarks → 40 features (x, y each)
KEY_LANDMARK_IDS = [
    4,    # nose tip
    152,  # chin
    33,   # left eye inner corner
    263,  # right eye inner corner
    130,  # left eye outer corner
    359,  # right eye outer corner
    10,   # forehead top
    234,  # left ear
    454,  # right ear
    13,   # mouth centre
    61,   # left mouth corner
    291,  # right mouth corner
    6,    # nose bridge
    168,  # nose midpoint
    197,  # nose lower
    205,  # left cheek
    425,  # right cheek
    70,   # left brow
    300,  # right brow
    162,  # left temple
]

# ═══════════════════════════════════════════════════════════════════
#  MEDIAPIPE SETUP  (same FaceMesh config as main.py)
# ═══════════════════════════════════════════════════════════════════

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Error: could not open camera.")
    exit()

# ═══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (must match head_model.py exactly)
# ═══════════════════════════════════════════════════════════════════

def extract_features(face_landmarks):
    """
    40-element float32 array: (x, y) for each key landmark, translated
    to face-centre and scaled by face width.

    OPTIMISED: uses 4 fixed anchor landmarks (ears 234/454, forehead 10,
    chin 152) for O(4) normalization instead of iterating all 478 landmarks.
    Must match _extract_features() in head_model.py exactly.
    """
    lms   = face_landmarks.landmark
    l_ear = lms[234]; r_ear = lms[454]
    cx    = (l_ear.x + r_ear.x) / 2.0
    cy    = (lms[10].y + lms[152].y) / 2.0
    scale = abs(r_ear.x - l_ear.x)
    if scale < 1e-6:
        return None

    features = []
    for idx in KEY_LANDMARK_IDS:
        lm = lms[idx]
        features.append((lm.x - cx) / scale)
        features.append((lm.y - cy) / scale)
    return np.array(features, dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════
#  HUD HELPERS
# ═══════════════════════════════════════════════════════════════════

def draw_centered(frame, text, y, scale=0.9, color=(255, 255, 255), thickness=2):
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (frame.shape[1] - w) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def dark_panel(frame, y1=70, y2=420):
    overlay = frame.copy()
    cv2.rectangle(overlay, (30, y1), (610, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

def quit_check():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit — no model saved.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ═══════════════════════════════════════════════════════════════════
#  DATA COLLECTION LOOP
# ═══════════════════════════════════════════════════════════════════

all_features: list = []
all_labels:   list = []

total_dirs = ROUNDS * len(DIRECTIONS)
completed  = 0

print(f"Collecting {ROUNDS} rounds × {len(DIRECTIONS)} directions × "
      f"{COLLECT_SECONDS}s each.")
print("Press Q at any time to quit without saving.\n")

for round_idx in range(ROUNDS):
    round_num    = round_idx + 1
    round_label  = f"Round {round_num} of {ROUNDS}"
    round_hint   = ("Hold still" if round_idx == 0
                    else "Vary position within the direction")
    prompt_map   = ROUND_PROMPTS[round_idx]

    # Brief inter-round title card
    title_end = time.time() + 2.5
    while time.time() < title_end:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        dark_panel(frame, 120, 360)
        draw_centered(frame, round_label,   200, scale=1.3,
                      color=(0, 200, 120), thickness=3)
        draw_centered(frame, round_hint,    255, scale=0.75,
                      color=(190, 190, 190))
        draw_centered(frame, "Starting in a moment...", 305,
                      scale=0.7, color=(160, 160, 160))
        cv2.imshow("FacePlay — Training", frame)
        quit_check()

    for direction in DIRECTIONS:
        prompt = prompt_map[direction]
        completed += 1
        progress   = f"{completed}/{total_dirs}"
        print(f"\n  {progress}  [{direction}]  {prompt}")

        # ── Countdown ──────────────────────────────────────────────
        countdown_end = time.time() + COUNTDOWN_SECONDS
        while time.time() < countdown_end:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            remaining = countdown_end - time.time()

            dark_panel(frame)
            draw_centered(frame, f"{round_label}  ({progress})",
                          105, scale=0.65, color=(120, 120, 120))
            draw_centered(frame, direction.replace("_", " "),
                          175, scale=1.4, color=(0, 200, 120), thickness=3)
            draw_centered(frame, prompt,
                          230, scale=0.65, color=(190, 190, 190))
            draw_centered(frame, f"Get ready  {remaining:.1f}s",
                          305, scale=1.0, color=(255, 180, 0), thickness=2)

            cv2.imshow("FacePlay — Training", frame)
            quit_check()

        # ── Recording ──────────────────────────────────────────────
        collect_end      = time.time() + COLLECT_SECONDS
        frames_collected = 0

        while time.time() < collect_end:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            remaining = collect_end - time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)

            got_face = False
            if results.multi_face_landmarks:
                feats = extract_features(results.multi_face_landmarks[0])
                if feats is not None:
                    all_features.append(feats)
                    all_labels.append(direction)
                    frames_collected += 1
                    got_face = True

            dark_panel(frame)
            draw_centered(frame, f"{round_label}  ({progress})",
                          105, scale=0.65, color=(120, 120, 120))
            draw_centered(frame, direction.replace("_", " "),
                          165, scale=1.4, color=(0, 200, 120), thickness=3)
            draw_centered(frame, prompt,
                          218, scale=0.65, color=(190, 190, 190))
            rec_color = (0, 220, 0) if got_face else (0, 60, 220)
            draw_centered(frame, "● RECORDING",
                          278, scale=1.0, color=rec_color, thickness=2)
            draw_centered(frame, f"{remaining:.1f}s  |  {frames_collected} frames",
                          323, scale=0.7, color=(200, 200, 200))
            if not got_face:
                draw_centered(frame, "NO FACE — move closer or improve lighting",
                              368, scale=0.6, color=(0, 80, 255))

            cv2.imshow("FacePlay — Training", frame)
            quit_check()

        print(f"     Collected {frames_collected} frames.")

cap.release()
cv2.destroyAllWindows()

# ═══════════════════════════════════════════════════════════════════
#  DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════
# Each real frame spawns AUG_COPIES jittered copies.
# The noise is tiny (< 0.5% of face width) — it doesn't change the
# meaning of any sample, it just prevents the model from memorising
# exact landmark positions and forces it to learn ranges instead.

real_count = len(all_features)
print(f"\nReal frames collected: {real_count}")
print(f"Augmenting ×{AUG_COPIES} with σ={AUG_NOISE_STD} noise...")

aug_features = list(all_features)
aug_labels   = list(all_labels)

rng = np.random.default_rng(seed=42)
for feat, label in zip(all_features, all_labels):
    noise = rng.normal(0, AUG_NOISE_STD,
                       size=(AUG_COPIES, len(feat))).astype(np.float32)
    for n in noise:
        aug_features.append(feat + n)
        aug_labels.append(label)

print(f"Total samples after augmentation: {len(aug_features)}")

# ═══════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════

print(f"\nTraining RBF-SVM on {len(aug_features)} samples "
      f"({len(set(aug_labels))} classes)...")

X = np.array(aug_features)
y = np.array(aug_labels)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# RBF-SVM: non-linear kernel handles the close boundaries between
# UP / UP_LEFT / UP_RIGHT far better than a linear classifier.
# probability=True enables predict_proba() so main.py can gate on
# confidence and ignore low-certainty frames.
clf = SVC(kernel='rbf', C=10.0, gamma='scale',
          probability=True, random_state=42)
clf.fit(X_scaled, y)

# Cross-validate on the REAL frames only so augmented copies don't
# inflate the score (they are very similar to training samples).
X_real   = scaler.transform(np.array(all_features))
y_real   = np.array(all_labels)
scores   = cross_val_score(clf, X_real, y_real, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy (real frames only): "
      f"{scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

if scores.mean() < 0.88:
    print("\nTips for higher accuracy:")
    print("  • Re-run — hold each position more steadily in Round 1")
    print("  • Make sure your face is fully in frame and well-lit")
    print("  • Increase COLLECT_SECONDS or ROUNDS at the top of this file")
elif scores.mean() >= 0.95:
    print("Excellent accuracy — model should feel very responsive.")
else:
    print("Good accuracy. Run main.py and test.")

# ═══════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════

with open(MODEL_FILE, 'wb') as f:
    pickle.dump({
        'model':        clf,
        'scaler':       scaler,
        'landmark_ids': KEY_LANDMARK_IDS,
    }, f)

print(f"\nSaved → {MODEL_FILE}")
print("You can now run main.py.")
