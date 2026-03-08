import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
import speech_recognition as sr
from collections import deque
import head_model

# ═══════════════════════════════════════════════════════════════════
#  FacePlay — Hands-free game controller
#  Built for hospital patients and anyone who cannot use their hands.
#  A standard webcam is all that is needed.
#
#  Python 3.11 | mediapipe==0.10.14
# ═══════════════════════════════════════════════════════════════════

# ── PyAutoGUI optimization ───────────────────────────────────────
pyautogui.PAUSE    = 0      # remove the default 0.1 s delay between every call
pyautogui.FAILSAFE = False  # disable the move-to-corner kill switch

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION — all tunable constants live here, no magic numbers
# ═══════════════════════════════════════════════════════════════════

# ── Camera ───────────────────────────────────────────────────────
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAM_FPS    = 60

# ── Startup calibration ──────────────────────────────────────────
CALIB_DURATION    = 2.0   # seconds to collect baseline face data
WINK_CALIB_RATIO  = 0.75  # wink threshold = this fraction of neutral EAR

# ── Wink / blink (EAR) ───────────────────────────────────────────
WINK_THRESHOLD   = 0.12  # fallback if calibration finds no face
# CHANGED: replaced DEBOUNCE_FRAMES (frame-count) with WINK_DEBOUNCE_SEC (seconds).
# Frame counting is unreliable because FPS varies — at 30 fps, 5 frames = 167 ms;
# at 60 fps, 5 frames = 83 ms. A fixed time value gives consistent feel regardless
# of camera speed. 0.25 s means the key can fire at most 4 times per second,
# and a single closed eye never re-fires while still shut.
WINK_DEBOUNCE_SEC = 0.25  # seconds to lock out after a wink/blink fires

# ── Eyebrow landmark geometry ────────────────────────────────────
BROW_YAW_THRESHOLD    = 8     # degrees of yaw to count as "turned"
BROW_SUSTAIN_FRAMES   = 2     # consecutive frames above trigger before hold fires

# Multipliers applied to the neutral brow ratio measured during calibration.
# Trigger = 15 % above neutral; release = 5 % above neutral.
# Adjust these two values to tune sensitivity without touching the math.
BROW_TRIGGER_MULTIPLIER = 1.15
BROW_RELEASE_MULTIPLIER = 1.05

# Asymmetric trigger / release thresholds (geometry-based ratios).
# Set from NEUTRAL_BROW_RATIO * multiplier during calibration.
# Fallback values assume a neutral ratio of ~0.35 (nose-bridge formula).
BROW_TRIGGER_THRESHOLD = 0.403  # ratio to start holding (head forward)
BROW_RELEASE_THRESHOLD = 0.368  # ratio to release — lower = earlier release
# Head-turned variants: +0.07 on trigger, release keeps 0.035 gap
BROW_TRIGGER_TURNED    = 0.473  # trigger when head is rotated (stricter)
BROW_RELEASE_TURNED    = 0.438  # release when head is rotated

# CHANGED: velocity-based short-hop detection.
# A fast upward flick of the brows (high velocity) fires a timed press instead
# of keyDown/Up, giving a short hop.  A slow sustained raise uses the existing
# hold system for a tall jump.  This makes short hops feel natural and distinct.
BROW_VELOCITY_THRESHOLD = 0.08  # score-change-per-frame to count as a "fast flick"
SHORT_HOP_DURATION_MS   = 80    # milliseconds to hold Z for a short hop

# ── Head pose / LEFT-RIGHT hysteresis ────────────────────────────
# CHANGED: HORIZ_ENTER raised from 7 → 10 so the neutral dead zone is wider.
# A bigger turn is now required to leave FORWARD and enter LEFT or RIGHT,
# which means small head wobbles near center no longer re-trigger movement.
# HORIZ_EXIT stays at 3 — it is still easy to return to neutral once moving.
HORIZ_ENTER = 10  # yaw degrees to activate LEFT or RIGHT (wider neutral zone)
HORIZ_EXIT  = 3   # yaw degrees to release back to FORWARD (unchanged)

# ── ML confidence gate ───────────────────────────────────────────
# The RBF-SVM returns a probability for each class.  If the winning
# class probability is below this threshold the frame is ambiguous
# (head is between two positions) and the previous direction is kept.
# This eliminates the jitter that happens when the model oscillates
# between two close classes during a transition.
# Range: 0.0 (accept everything) → 1.0 (only rock-solid predictions).
# 0.55 is a good starting point; raise if still jittery.
DIRECTION_MIN_CONFIDENCE = 0.55

# ── No-face auto-pause ───────────────────────────────────────────
NO_FACE_PAUSE_DELAY = 3.0  # seconds with no face before auto-pressing ESC

# ── HUD feedback timers ──────────────────────────────────────────
ACTION_FEEDBACK_DUR = 0.4  # seconds to show blink/wink action label on HUD

# ── Debug overlay ────────────────────────────────────────────────
# Set True to show raw EAR values and head-angle numbers on screen.
DEBUG_MODE = False

# ═══════════════════════════════════════════════════════════════════
#  KEY MAPPINGS
# ═══════════════════════════════════════════════════════════════════
# CHANGED: "RightWink" renamed to "Blink" — attack now requires both eyes.
# CHANGED: Interact eyebrow mapped to 'i' (inventory, single press).
#          Attack eyebrow mapped to 'z' (jump, held key).
keys = {
    "Attack": {
        "Blink":    'x',  # full blink (both eyes)  → attack
        "LeftWink": 'c',  # left wink only          → dash
        "Eyebrow":  'z',  # eyebrow raise           → hold jump
    },
    "Interact": {
        "Blink":   'e',   # full blink → interact / confirm
        "Eyebrow": 'i',   # eyebrow raise → press inventory (single press, not hold)
        # LeftWink intentionally absent in Interact mode
    }
}

# Head movement keys differ per mode.
# Attack  → arrow keys (Hollow Knight)
# Interact → WASD     (Stardew / Undertale)
head_keys = {
    "Attack"  : {"left": "left", "right": "right", "up": "up",  "down": "down"},
    "Interact": {"left": "a",    "right": "d",      "up": "w",   "down": "s"}
}

# ═══════════════════════════════════════════════════════════════════
#  MEDIAPIPE — FaceLandmarker  (winks + eyebrows)
# ═══════════════════════════════════════════════════════════════════
# Uses the task-based API with blendshapes.
# Model file must sit in the same folder as this script.

BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker     = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

MODEL_PATH = 'face_landmarker.task'
fl_options = FaceLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)
detector = FaceLandmarker.create_from_options(fl_options)

# EAR landmark indices (unchanged from original)
LEFT_EAR_IDS  = [33, 159, 158, 133, 153, 145]
RIGHT_EAR_IDS = [362, 380, 374, 263, 386, 385]

# Brow-raise geometry landmark indices
# FIX: eye-top landmarks (159, 386) were the old reference points.
# When the eye closes during a blink the eyelid moves UP toward the
# brow, shrinking the brow-to-eye distance — indistinguishable from
# an actual brow raise.  The nose bridge (6) and chin (152) are
# completely rigid during blinks and expressions, so using them as
# reference and normalization anchors fully separates blink and
# eyebrow detection.
LEFT_BROW_IDS    = [70, 63, 105, 66, 107]   # five left-brow points
RIGHT_BROW_IDS   = [336, 296, 334, 293, 300] # five right-brow points
NOSE_BRIDGE      = 6    # top of nose bridge — stable reference, never moves during blinks
FACE_HEIGHT_TOP  = 10   # top of forehead   — for normalization
FACE_HEIGHT_BOT  = 152  # chin              — for normalization

# Pre-computed solvePnP constants — kept for get_head_direction rollback only.
# FaceMesh is no longer instantiated; head direction is handled by head_model.
LANDMARK_IDS = {1, 33, 61, 199, 263, 291}
dist_matrix  = np.zeros((4, 1), dtype=np.float64)

# ── Hysteresis direction sets (module-level — no per-frame allocation) ──
_LEFT_DIRS  = frozenset({"LEFT"})
_RIGHT_DIRS = frozenset({"RIGHT"})
_DIAGONALS  = frozenset({"UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT"})

# ═══════════════════════════════════════════════════════════════════
#  CAMERA SETUP
# ═══════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

if not cap.isOpened():
    print("FacePlay: could not open camera — exiting.")
    exit()

# ═══════════════════════════════════════════════════════════════════
#  SHARED STATE  (read/written by both camera loop and speech thread)
# ═══════════════════════════════════════════════════════════════════
# Both ControlMode and is_paused share a single lock for simplicity.
state_lock  = threading.Lock()
ControlMode = "Attack"  # "Attack" or "Interact"
is_paused   = False     # True while the game is paused

# ═══════════════════════════════════════════════════════════════════
#  GESTURE STATE  (camera loop only — no locking needed)
# ═══════════════════════════════════════════════════════════════════

# ── Wink / blink debounce ────────────────────────────────────────
# CHANGED: replaced bool+frame-counter pairs with a single unlock timestamp
# per eye. leftWinkUntil = the wall-clock time after which the left eye is
# allowed to fire again. 0.0 means "ready now". No frame counting needed.
leftWinkUntil  = 0.0
rightWinkUntil = 0.0

# ── Eyebrow hold state ───────────────────────────────────────────
last_brow_state    = False  # True while jump key is held down (Attack mode only)
brow_sustain_count = 0      # consecutive up-frames; fires at BROW_SUSTAIN_FRAMES
# CHANGED: previous frame's weighted brow score — used to compute velocity
# (how fast the brow is moving). Reset to 0 when face is lost so velocity
# doesn't carry over from a previous detection window.
prev_brow_ratio    = 0.0
# Tracks whether a short-hop timed press is currently in flight.
# Stores the time.time() when the key should be released, or 0 if idle.
short_hop_release_at = 0.0

# ── Head movement ────────────────────────────────────────────────
direction_buffer = deque(maxlen=4)  # kept for import; no longer used for UP/DOWN
last_direction   = None
horiz_active     = None   # "LEFT", "RIGHT", or None — managed by hysteresis
last_head_yaw    = 0.0    # yaw from previous frame (for adaptive brow threshold)

# ── No-face tracking ─────────────────────────────────────────────
face_lost_time = None   # wall-clock time when face disappeared; None if present
auto_paused    = False  # True if we already auto-pressed ESC for this disappearance

# ── HUD action feedback ──────────────────────────────────────────
action_text        = ""   # e.g. "ATTACKING", "DASHING"
action_text_until  = 0.0  # show action_text until this timestamp

# ── Mode-change tracking (camera loop internal) ──────────────────
prev_mode   = "Attack"
prev_paused = False

# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def EAR(face, ids):
    """Eye Aspect Ratio — lower = more closed. Unchanged from original."""
    p2p6 = abs(face[ids[1]].y - face[ids[5]].y)
    p3p5 = abs(face[ids[2]].y - face[ids[4]].y)
    p1p4 = abs(face[ids[0]].x - face[ids[3]].x)
    return (p2p6 + p3p5) / (2.0 * p1p4)


def get_brow_raise(face):
    """
    Compute brow-raise ratio using nose bridge as the stable reference.

    FIX: the previous version measured (eye_top_y - brow_y) / face_height.
    Eye-top landmarks move UP during a blink (eyelid closing), making the
    brow-to-eye distance shrink — identical to an actual eyebrow raise.
    This caused a false jump on every blink.

    The nose bridge (NOSE_BRIDGE) is completely rigid during blinks and
    facial expressions.  Measuring how far the brow sits ABOVE the nose
    bridge means blinks have zero effect on the ratio — only a real
    eyebrow raise changes the value.

    When brows raise: brow_y decreases (moves toward top of screen)
                      → (nose_bridge_y - brow_y) increases → ratio goes up
    When eyes blink:  brow_y is unchanged → ratio stays flat
    """
    face_height = abs(face[FACE_HEIGHT_TOP].y - face[FACE_HEIGHT_BOT].y)
    if face_height < 1e-6:
        return 0.0

    left_brow_y  = sum(face[i].y for i in LEFT_BROW_IDS)  / len(LEFT_BROW_IDS)
    right_brow_y = sum(face[i].y for i in RIGHT_BROW_IDS) / len(RIGHT_BROW_IDS)
    avg_brow_y   = (left_brow_y + right_brow_y) / 2.0
    nose_bridge_y = face[NOSE_BRIDGE].y
    return (nose_bridge_y - avg_brow_y) / face_height


def get_head_direction(face_landmarks, img_w, img_h, focal_length, cam_matrix):
    """
    Estimate head orientation using solvePnP on 6 facial landmarks.
    Returns (direction_string, pitch_degrees, yaw_degrees).
    Unchanged from original head-movement.py.
    """
    face_3d, face_2d = [], []
    found = 0
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in LANDMARK_IDS:
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
            found += 1
            if found == 6:
                break

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360

    if   x > 12   and y > 10:    return "UP_RIGHT",   x, y
    elif x > 15.5 and y < -11.5: return "UP_LEFT",    x, y
    elif x < 1    and y > 6:     return "DOWN_RIGHT",  x, y
    elif x < 1    and y < -6:    return "DOWN_LEFT",   x, y
    elif y < -7:                  return "LEFT",        x, y
    elif y > 6:                   return "RIGHT",       x, y
    elif x < -3:                  return "DOWN",        x, y
    elif x > 12:                  return "UP",          x, y
    else:                         return "FORWARD",     x, y


# ── Diagonal sequential-press delay ─────────────────────────────
# When a diagonal is detected, FacePlay fires key1 then key2 this many
# milliseconds apart — the "quick succession" pattern used in game combos.
# e.g. DOWN_LEFT = press DOWN → 60 ms later → press LEFT.
DIAGONAL_PRESS_DELAY_MS = 60


def _fire_successive(key1, key2):
    """
    Press key1, wait DIAGONAL_PRESS_DELAY_MS, then press key2.
    Runs in a daemon thread so the camera loop is never blocked.
    """
    def _run():
        pyautogui.press(key1)
        time.sleep(DIAGONAL_PRESS_DELAY_MS / 1000.0)
        pyautogui.press(key2)
    threading.Thread(target=_run, daemon=True).start()


def send_keys(direction, mode):
    """
    Fire keyDown/keyUp for a head-pose direction.
    LEFT/RIGHT use the hysteresis system — not handled here.
    Diagonals fire as two successive single presses (key1 then key2),
    one-shot per detection (fires once when the diagonal is first entered).
    Mode-aware: Attack → arrow keys, Interact → WASD.
    """
    k = head_keys.get(mode, head_keys["Attack"])
    L, R, U, D = k["left"], k["right"], k["up"], k["down"]

    if direction == "UP":
        pyautogui.keyDown(U); pyautogui.keyUp(D); pyautogui.keyUp(L); pyautogui.keyUp(R)
    elif direction == "DOWN":
        pyautogui.keyDown(D); pyautogui.keyUp(U); pyautogui.keyUp(L); pyautogui.keyUp(R)
    elif direction == "UP_LEFT":
        pyautogui.keyUp(D); pyautogui.keyUp(R); _fire_successive(U, L)
    elif direction == "UP_RIGHT":
        pyautogui.keyUp(D); pyautogui.keyUp(L); _fire_successive(U, R)
    elif direction == "DOWN_LEFT":
        pyautogui.keyUp(U); pyautogui.keyUp(R); _fire_successive(D, L)
    elif direction == "DOWN_RIGHT":
        pyautogui.keyUp(U); pyautogui.keyUp(L); _fire_successive(D, R)
    elif direction == "FORWARD":
        pyautogui.keyUp(L); pyautogui.keyUp(R); pyautogui.keyUp(U); pyautogui.keyUp(D)


def release_all_keys():
    """
    Release every key FacePlay can hold.
    Called on mode switch, face loss, pause, and quit.
    Covers both Arrow and WASD sets so nothing gets stuck on mode change.
    """
    for key in ("left", "right", "up", "down", "a", "d", "w", "s", "z"):
        pyautogui.keyUp(key)


# ═══════════════════════════════════════════════════════════════════
#  SPEECH RECOGNITION THREAD
# ═══════════════════════════════════════════════════════════════════
# Runs as a daemon thread so it never blocks the camera loop.
# Writes to ControlMode and is_paused under state_lock.

def speech_thread_fn():
    global ControlMode, is_paused

    r = sr.Recognizer()
    r.energy_threshold         = 300    # minimum volume to count as speech
    r.pause_threshold          = 0.5    # seconds of silence = phrase end
    r.dynamic_energy_threshold = False  # no auto-adjust (can cause missed words)

    key_phrases = ["pause", "resume", "fight", "interact"]
    last_said   = ""

    try:
        print("Speech: calibrating to room noise...")
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
        print(f"Speech: ready. Listening for: {', '.join(key_phrases)}")
    except Exception as e:
        print(f"Speech: microphone error during calibration — {e}")
        return  # exit thread gracefully; program continues without voice

    while True:
        try:
            with sr.Microphone() as source:
                try:
                    audio = r.listen(source, timeout=3, phrase_time_limit=3)
                except sr.WaitTimeoutError:
                    continue

            text = r.recognize_google(audio).lower()
            print(f"Speech heard: {text}")

            if "pause" in text and last_said != "pause":
                with state_lock:
                    is_paused = True
                pyautogui.press('escape')
                last_said = "pause"

            elif "resume" in text and last_said != "resume":
                # CHANGED: is_paused set to False (original had no is_paused)
                with state_lock:
                    is_paused = False
                pyautogui.press('escape')
                last_said = "resume"

            elif "fight" in text:
                with state_lock:
                    ControlMode = "Attack"
                release_all_keys()
                print("COMMAND: switched to Attack mode")

            elif "interact" in text:
                with state_lock:
                    ControlMode = "Interact"
                release_all_keys()
                print("COMMAND: switched to Interact mode")

        except sr.UnknownValueError:
            pass   # could not understand — ignore quietly
        except sr.RequestError as e:
            print(f"Speech: Google API error — {e}")
        except Exception as e:
            print(f"Speech: unexpected error — {e}")
            time.sleep(0.5)  # brief pause to avoid tight crash-loop


speech_thread = threading.Thread(target=speech_thread_fn, daemon=True)
speech_thread.start()

# ═══════════════════════════════════════════════════════════════════
#  STARTUP CALIBRATION PHASE
# ═══════════════════════════════════════════════════════════════════
# Shows a welcome screen while collecting CALIB_DURATION seconds of
# the user's neutral face data. Computes personalised EAR and brow
# thresholds from their baseline so the controller works for any face.

ear_samples  = []
brow_samples = []
calib_start  = time.time()

print("FacePlay: calibrating to your face — please look at the camera.")

while time.time() - calib_start < CALIB_DURATION:
    ret, frame = cap.read()
    if not ret:
        continue

    frame  = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    # Run FaceLandmarker to collect baseline samples
    try:
        ts         = int(time.time() * 1000)
        fl_results = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), ts)

        if fl_results.face_landmarks:
            face  = fl_results.face_landmarks[0]
            l_ear = EAR(face, LEFT_EAR_IDS)
            r_ear = EAR(face, RIGHT_EAR_IDS)
            ear_samples.append((l_ear + r_ear) / 2.0)

            brow_samples.append(get_brow_raise(face))
    except Exception:
        pass  # skip bad frames silently during calibration

    # ── Welcome screen HUD ───────────────────────────────────────
    remaining = max(0.0, CALIB_DURATION - (time.time() - calib_start))

    # Semi-transparent dark panel so text is readable over any background
    overlay = frame.copy()
    cv2.rectangle(overlay, (50, 130), (590, 360), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "Welcome to FacePlay",
                (90, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(frame, "Sit comfortably and look at the camera",
                (62, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1)
    cv2.putText(frame, f"Calibrating  {remaining:.1f}s",
                (195, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 210, 120), 2)

    cv2.imshow("FacePlay", frame)
    cv2.waitKey(1)

# ── Compute personalised thresholds from collected samples ───────
if ear_samples:
    neutral_ear    = sum(ear_samples) / len(ear_samples)
    WINK_THRESHOLD = neutral_ear * WINK_CALIB_RATIO
    print(f"Calibration: neutral EAR={neutral_ear:.3f}  "
          f"→ wink threshold={WINK_THRESHOLD:.3f}")
else:
    print("Calibration: no face found — using default wink threshold "
          f"({WINK_THRESHOLD})")

if brow_samples:
    neutral_brow           = sum(brow_samples) / len(brow_samples)
    # Personalise thresholds as multiples of the neutral geometry ratio.
    # Trigger = 25 % above neutral; release = 10 % above neutral.
    # The gap between the two gives asymmetric hysteresis so keyUp fires
    # well before the brows fully relax (eliminating hold-on-release delay).
    # Turned variants add a fixed +0.07 to the trigger (stricter when
    # the head is rotated because geometry shifts upward slightly).
    BROW_TRIGGER_THRESHOLD = neutral_brow * BROW_TRIGGER_MULTIPLIER
    BROW_RELEASE_THRESHOLD = neutral_brow * BROW_RELEASE_MULTIPLIER
    BROW_TRIGGER_TURNED    = BROW_TRIGGER_THRESHOLD + 0.07
    BROW_RELEASE_TURNED    = BROW_TRIGGER_TURNED - 0.035
    print(f"Calibration: neutral brow ratio={neutral_brow:.3f}  "
          f"→ trigger={BROW_TRIGGER_THRESHOLD:.3f} / release={BROW_RELEASE_THRESHOLD:.3f} "
          f"(turned: {BROW_TRIGGER_TURNED:.3f} / {BROW_RELEASE_TURNED:.3f})")
else:
    print("Calibration: no face found — using default brow thresholds "
          f"(trigger={BROW_TRIGGER_THRESHOLD} / release={BROW_RELEASE_THRESHOLD})")

# ── Show "Calibrated!" confirmation for 1 second ─────────────────
confirm_start = time.time()
while time.time() - confirm_start < 1.0:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (50, 150), (590, 330), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "Calibrated!",
                (175, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 210, 120), 3)
    cv2.putText(frame, "Starting in Attack Mode",
                (130, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 190, 190), 2)

    cv2.imshow("FacePlay", frame)
    cv2.waitKey(1)

# ── Load ML head-direction model ─────────────────────────────────
try:
    head_model.load()
except FileNotFoundError as e:
    print(e)
    print("FacePlay: no head_model.pkl — head movement will be disabled.")

print("FacePlay: entering main loop — press Q to quit.")

# ═══════════════════════════════════════════════════════════════════
#  MAIN CAMERA LOOP
# ═══════════════════════════════════════════════════════════════════
while True:

    ret, frame = cap.read()
    if not ret:
        print("FacePlay: failed to grab frame.")
        break

    start = time.time()

    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]

    # ── Snapshot shared state once per frame (single lock acquire) ─
    with state_lock:
        current_mode   = ControlMode
        current_paused = is_paused

    # ── Detect mode or pause transitions ─────────────────────────
    # When mode changes: release all keys and reset movement state so
    # nothing from the old mode bleeds into the new one.
    if current_mode != prev_mode:
        release_all_keys()
        horiz_active       = None
        last_direction     = None
        last_brow_state    = False
        brow_sustain_count = 0
        prev_mode = current_mode

    # When unpausing: reset face-loss tracking so we don't immediately
    # re-trigger the no-face countdown from before the pause.
    if prev_paused and not current_paused:
        face_lost_time = None
        auto_paused    = False
    prev_paused = current_paused

    # ── Prepare RGB frame shared by both detectors ───────────────
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    # ── PAUSED: show overlay, skip all gesture detection ─────────
    if current_paused:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, img_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, "PAUSED",
                    (img_w // 2 - 105, img_h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 50, 220), 4)
        cv2.putText(frame, 'Say "resume" to continue',
                    (img_w // 2 - 158, img_h // 2 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 190, 190), 2)

        totalTime = time.time() - start
        fps = 1 / totalTime if totalTime > 0 else 0
        cv2.putText(frame, f"FPS: {int(fps)}", (10, img_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1)

        cv2.imshow("FacePlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ═══════════════════════════════════════════════════════════════
    #  DETECTOR 1 — FaceLandmarker  (winks / blink / eyebrows)
    # ═══════════════════════════════════════════════════════════════
    active_gesture    = ""     # updated below; shown on HUD
    leftValue         = 0.0    # initialise so DEBUG_MODE block is always safe
    rightValue        = 0.0
    current_brow_ratio = 0.0   # shown on HUD; updated when face is detected

    try:
        timestamp_ms = int(time.time() * 1000)
        fl_results   = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame),
            timestamp_ms
        )
    except Exception as e:
        print(f"FaceLandmarker error: {e}")
        fl_results = None

    face_detected = fl_results is not None and bool(fl_results.face_landmarks)

    if face_detected:
        # Face is present — reset the no-face timer if it was running
        if face_lost_time is not None:
            face_lost_time = None
            auto_paused    = False
            release_all_keys()

        face       = fl_results.face_landmarks[0]
        leftValue  = EAR(face, LEFT_EAR_IDS)
        rightValue = EAR(face, RIGHT_EAR_IDS)

        # ── Blink / wink detection ────────────────────────────────
        # Priority: full blink (both eyes) is checked FIRST so it never
        # accidentally also fires a left-wink on the same frame.
        #
        # 1. Both EARs low → full blink → attack / interact
        # 2. Left EAR only low → left wink → dash
        # 3. Right EAR only low → intentionally ignored
        bothClosed = leftValue < WINK_THRESHOLD and rightValue < WINK_THRESHOLD

        # CHANGED: debounce is now time-based. Each eye has an "unlock time" —
        # a timestamp after which it is allowed to fire again. Checking
        # `now >= xWinkUntil` means the eye is ready; firing sets the unlock
        # time to `now + WINK_DEBOUNCE_SEC`, blocking re-fires for that window
        # regardless of how long the eye stays closed or what the frame rate is.
        now = time.time()

        if bothClosed and now >= leftWinkUntil and now >= rightWinkUntil:
            # ── Full blink → attack (Attack) / interact (Interact) ─
            # Lock both eyes out so neither wink branch re-fires while
            # the eyes are still closed.
            leftWinkUntil  = now + WINK_DEBOUNCE_SEC
            rightWinkUntil = now + WINK_DEBOUNCE_SEC
            res = keys.get(current_mode, {}).get("Blink")
            if res:
                pyautogui.press(res)
            action_text       = "ATTACKING" if current_mode == "Attack" else "INTERACTING"
            action_text_until = now + ACTION_FEEDBACK_DUR

        else:
            # ── Left wink only → dash ─────────────────────────────
            # Only reachable when bothClosed is False, so a blink
            # never triggers a dash at the same time.
            if leftValue < WINK_THRESHOLD and now >= leftWinkUntil:
                leftWinkUntil = now + WINK_DEBOUNCE_SEC
                res = keys.get(current_mode, {}).get("LeftWink")
                if res:
                    pyautogui.press(res)
                action_text       = "DASHING"
                action_text_until = now + ACTION_FEEDBACK_DUR

            # right-only wink intentionally ignored

        # No cooldown tick needed — time.time() handles expiry automatically.

        # ── Eyebrow raise ─────────────────────────────────────────
        # Raw geometry: (eye_top_y - brow_avg_y) / face_height.
        # Larger value = brows raised higher relative to the eyes.
        weighted_avg       = get_brow_raise(face)
        current_brow_ratio = weighted_avg   # expose to HUD

        # Select trigger/release pair based on head rotation.
        # last_head_yaw is from the previous FaceMesh frame (one frame
        # behind at 60 fps — imperceptible lag).
        turned = abs(last_head_yaw) > BROW_YAW_THRESHOLD
        brow_trigger  = BROW_TRIGGER_TURNED   if turned else BROW_TRIGGER_THRESHOLD
        brow_release  = BROW_RELEASE_TURNED   if turned else BROW_RELEASE_THRESHOLD

        # Asymmetric brow_up / brow_down flags.
        # brow_up uses the higher trigger threshold (ratio must climb
        # before keyDown fires). brow_down uses the lower release threshold
        # (ratio only needs to drop partway before keyUp fires).
        brow_up   = weighted_avg > brow_trigger
        brow_down = weighted_avg < brow_release

        # Velocity = ratio change since last frame.
        # A fast upward flick produces a high positive velocity spike.
        brow_velocity   = weighted_avg - prev_brow_ratio
        prev_brow_ratio = weighted_avg   # store for next frame

        res = keys.get(current_mode, {}).get("Eyebrow")

        # ── Short-hop timed release (Attack mode only) ──────────
        # If a short-hop press is in flight, check if its timed window
        # has expired and release the key.
        if short_hop_release_at > 0 and time.time() >= short_hop_release_at:
            short_hop_release_at = 0.0
            if current_mode == "Attack" and res:
                pyautogui.keyUp(res)
            last_brow_state = False

        if brow_up:
            brow_sustain_count += 1

            if not last_brow_state and current_mode == "Attack":
                # Velocity path — fast upward flick → short hop.
                # Fires immediately (no sustain wait) because the velocity
                # spike itself is the noise filter: a brief geometry blip
                # won't produce both a high ratio AND a high velocity.
                # Schedules an automatic keyUp after SHORT_HOP_DURATION_MS.
                if (brow_velocity >= BROW_VELOCITY_THRESHOLD
                        and short_hop_release_at == 0.0):
                    if res:
                        pyautogui.keyDown(res)
                    short_hop_release_at = time.time() + SHORT_HOP_DURATION_MS / 1000.0
                    last_brow_state      = True
                    brow_sustain_count   = 0  # reset so hold path doesn't double-fire

                # Hold path — slow sustained raise → tall jump.
                # Requires BROW_SUSTAIN_FRAMES consecutive frames before
                # firing so noise spikes that don't pass velocity never trigger.
                elif (brow_sustain_count >= BROW_SUSTAIN_FRAMES
                        and short_hop_release_at == 0.0):
                    last_brow_state = True
                    if res:
                        pyautogui.keyDown(res)

            elif not last_brow_state and current_mode != "Attack":
                # Interact mode: single press after sustain (unchanged)
                if brow_sustain_count >= BROW_SUSTAIN_FRAMES:
                    last_brow_state = True
                    if res:
                        pyautogui.press(res)

        # Release uses brow_down (lower threshold) not brow_up.
        # keyUp fires as soon as the ratio drops below BROW_RELEASE_THRESHOLD,
        # which happens well before it drops below BROW_TRIGGER_THRESHOLD.
        # No sustain counter on release — instant, same frame.
        # Short-hop timed presses manage their own release above; only the
        # hold path (short_hop_release_at == 0) needs explicit keyUp here.
        if brow_down:
            brow_sustain_count = 0
            if last_brow_state and short_hop_release_at == 0.0:
                last_brow_state = False
                if current_mode == "Attack" and res:
                    pyautogui.keyUp(res)

        if last_brow_state:
            active_gesture = "JUMPING" if current_mode == "Attack" else "INVENTORY"

    else:
        # ── No face detected ─────────────────────────────────────
        if face_lost_time is None:
            # Face just disappeared — release everything immediately
            face_lost_time     = time.time()
            auto_paused        = False
            last_brow_state      = False
            brow_sustain_count   = 0
            prev_brow_ratio      = 0.0   # reset so velocity doesn't carry over
            short_hop_release_at = 0.0   # cancel any in-flight short-hop timer
            horiz_active         = None
            last_direction     = None
            release_all_keys()

        seconds_missing = time.time() - face_lost_time

        # Auto-pause after NO_FACE_PAUSE_DELAY seconds — press ESC once only
        if seconds_missing >= NO_FACE_PAUSE_DELAY and not auto_paused:
            pyautogui.press('escape')
            auto_paused = True
            with state_lock:
                is_paused = True

    # ═══════════════════════════════════════════════════════════════
    #  HEAD DIRECTION  (ML model, reuses FaceLandmarker landmarks)
    # ═══════════════════════════════════════════════════════════════
    # FaceMesh is no longer run here.  The FaceLandmarker result from
    # Detector 1 already provides all 468 landmarks.  Passing the same
    # list to head_model.predict() eliminates one full model inference
    # per frame — the main source of lag.

    # ax/ay retained for DEBUG_MODE display compatibility.
    ax, ay = 0.0, 0.0

    last_confidence = 0.0   # reset each frame; used by debug overlay

    if face_detected:
        try:
            direction, approx_yaw, confidence = head_model.predict(
                fl_results.face_landmarks[0])
            ay              = approx_yaw
            last_confidence = confidence

            # Confidence gate — if the model is unsure (transitioning between
            # two positions) keep the previous committed direction instead of
            # firing an incorrect one.  FORWARD is always accepted so the
            # "release" path is never blocked.
            if confidence < DIRECTION_MIN_CONFIDENCE and direction != "FORWARD":
                direction = last_direction if last_direction is not None else "FORWARD"

        except Exception:
            direction = "FORWARD"

        last_head_yaw = ay  # feeds brow turned-threshold next frame

        # ── DIAGONALS — checked first, bypass horizontal hysteresis ──
        # All four diagonals fire as two quick successive key presses
        # (one-shot: fires once when the diagonal is first detected, not
        # every frame while held).  horiz_active is cleared so the
        # horizontal hysteresis doesn't also try to manage those keys.
        if direction in _DIAGONALS:
            if horiz_active is not None:
                horiz_active = None   # diagonal takes over; clear latched horiz
            if direction != last_direction:
                send_keys(direction, current_mode)
                last_direction = direction

        else:
            # ── LEFT / RIGHT hysteresis (pure horizontal only) ────
            new_horiz = horiz_active
            if horiz_active == "LEFT":
                if direction not in _LEFT_DIRS:
                    new_horiz = None
            elif horiz_active == "RIGHT":
                if direction not in _RIGHT_DIRS:
                    new_horiz = None
            else:
                if direction in _LEFT_DIRS:
                    new_horiz = "LEFT"
                elif direction in _RIGHT_DIRS:
                    new_horiz = "RIGHT"

            if new_horiz != horiz_active:
                horiz_active = new_horiz
                k = head_keys.get(current_mode, head_keys["Attack"])
                if horiz_active == "LEFT":
                    pyautogui.keyDown(k["left"]);  pyautogui.keyUp(k["right"])
                    # also release vertical in case we just left a diagonal
                    pyautogui.keyUp(k["up"]);      pyautogui.keyUp(k["down"])
                elif horiz_active == "RIGHT":
                    pyautogui.keyDown(k["right"]); pyautogui.keyUp(k["left"])
                    pyautogui.keyUp(k["up"]);      pyautogui.keyUp(k["down"])
                else:
                    pyautogui.keyUp(k["left"]);    pyautogui.keyUp(k["right"])
                last_direction = horiz_active if horiz_active else "FORWARD"

            # ── FORWARD: instant release ──────────────────────────
            if direction == "FORWARD":
                if last_direction != "FORWARD":
                    send_keys("FORWARD", current_mode)
                    last_direction = "FORWARD"

            # ── UP / DOWN: instant ────────────────────────────────
            elif direction not in ("LEFT", "RIGHT", *_DIAGONALS):
                if direction != last_direction:
                    send_keys(direction, current_mode)
                    last_direction = direction

        # ── Update active gesture text for HUD ────────────────────
        if active_gesture == "":
            if last_direction == "UP_LEFT":
                active_gesture = "FLICK UP → LEFT"
            elif last_direction == "UP_RIGHT":
                active_gesture = "FLICK UP → RIGHT"
            elif last_direction == "DOWN_LEFT":
                active_gesture = "FLICK DOWN → LEFT"
            elif last_direction == "DOWN_RIGHT":
                active_gesture = "FLICK DOWN → RIGHT"
            elif horiz_active == "LEFT":
                active_gesture = "MOVING LEFT"
            elif horiz_active == "RIGHT":
                active_gesture = "MOVING RIGHT"
            elif last_direction == "UP":
                active_gesture = "MOVING UP"
            elif last_direction == "DOWN":
                active_gesture = "MOVING DOWN"

    # ═══════════════════════════════════════════════════════════════
    #  HUD RENDERING
    # ═══════════════════════════════════════════════════════════════

    # ── Mode label — top left, large, colour-coded ────────────────
    # Orange for Attack (warm, action-oriented)
    # Blue for Interact (calm, exploration-oriented)
    mode_label = "ATTACK" if current_mode == "Attack" else "INTERACT"
    mode_color = (0, 120, 255) if current_mode == "Attack" else (200, 140, 0)
    cv2.putText(frame, mode_label, (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, mode_color, 3)

    # ── Active gesture — below mode label ─────────────────────────
    # Blink/wink feedback overrides movement text for ACTION_FEEDBACK_DUR s.
    now          = time.time()
    display_text = action_text if now < action_text_until else active_gesture
    if display_text:
        cv2.putText(frame, display_text, (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # ── Live brow ratio display (always on) ───────────────────────
    # Shows the raw geometry value and the current trigger threshold so
    # you can see exactly how close to firing the eyebrow is and tune
    # BROW_TRIGGER_MULTIPLIER without guessing.
    cv2.putText(frame, f"BROW: {current_brow_ratio:.3f}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, f"THRESHOLD: {BROW_TRIGGER_THRESHOLD:.3f}",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # ── No-face warning ───────────────────────────────────────────
    if face_lost_time is not None and not auto_paused:
        seconds_missing      = time.time() - face_lost_time
        remaining_to_pause   = max(0.0, NO_FACE_PAUSE_DELAY - seconds_missing)
        cx = img_w // 2
        cv2.putText(frame, "NO FACE DETECTED",
                    (cx - 155, img_h // 2 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 220), 2)
        cv2.putText(frame, f"Pausing in {remaining_to_pause:.1f}s",
                    (cx - 115, img_h // 2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 220), 2)

    # ── Debug overlay — disabled by default ──────────────────────
    if DEBUG_MODE:
        try:
            conf_color = (0, 240, 0) if last_confidence >= DIRECTION_MIN_CONFIDENCE \
                         else (0, 80, 255)
            cv2.putText(frame, f"conf:  {last_confidence:.2f}", (15, img_h - 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            cv2.putText(frame, f"pitch: {ax:.1f}", (15, img_h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 240), 1)
            cv2.putText(frame, f"yaw:   {ay:.1f}", (15, img_h - 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 240), 1)
            cv2.putText(frame, f"L EAR: {leftValue:.3f}", (15, img_h - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 0), 1)
            cv2.putText(frame, f"R EAR: {rightValue:.3f}", (15, img_h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 0), 1)
        except Exception:
            pass

    # ── FPS counter — bottom left ─────────────────────────────────
    totalTime = time.time() - start
    fps = 1 / totalTime if totalTime > 0 else 0
    cv2.putText(frame, f"FPS: {int(fps)}", (10, img_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)

    cv2.imshow("FacePlay", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ═══════════════════════════════════════════════════════════════════
#  CLEANUP
# ═══════════════════════════════════════════════════════════════════
release_all_keys()        # never leave a key held after quit
cap.release()
cv2.destroyAllWindows()
detector.close()
# speech thread is a daemon and exits automatically when the process ends
