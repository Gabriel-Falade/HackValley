import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
import queue
import speech_recognition as sr
from collections import deque
import head_model
import subprocess
import sys
import os
import ctypes

# Send a key directly to a window by title (bypasses focus requirement).
# Used for Flappy Bird so the space key reaches pygame even when the
# cv2 calibration window holds keyboard focus.
_WM_KEYDOWN = 0x0100
_WM_KEYUP   = 0x0101
_VK_SPACE   = 0x20
_VK_RETURN  = 0x0D

def _post_key_to_window(window_title, vk_code):
    """PostMessageW a keydown+keyup directly to the target window handle."""
    if sys.platform != 'win32':
        return
    hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
    if hwnd:
        ctypes.windll.user32.PostMessageW(hwnd, _WM_KEYDOWN, vk_code, 0)
        ctypes.windll.user32.PostMessageW(hwnd, _WM_KEYUP,   vk_code, 0)

def _post_key_partial(partial_title, vk_code):
    """Find a window whose title contains partial_title, post the key directly to it."""
    if sys.platform != 'win32':
        return
    found = []
    def _cb(hwnd, _):
        n = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, n + 1)
            if partial_title.lower() in buf.value.lower():
                found.append(hwnd)
        return True
    _EP = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_size_t, ctypes.c_size_t)
    ctypes.windll.user32.EnumWindows(_EP(_cb), 0)
    if found:
        ctypes.windll.user32.PostMessageW(found[0], _WM_KEYDOWN, vk_code, 0)
        ctypes.windll.user32.PostMessageW(found[0], _WM_KEYUP,   vk_code, 0)

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
WINK_THRESHOLD       = 0.12  # fallback full-blink threshold
LEFT_WINK_THRESHOLD  = 0.12  # fallback left-wink threshold (calibrated separately)
RIGHT_WINK_THRESHOLD = 0.12  # fallback right-wink threshold (calibrated separately)
# CHANGED: replaced DEBOUNCE_FRAMES (frame-count) with WINK_DEBOUNCE_SEC (seconds).
# Frame counting is unreliable because FPS varies — at 30 fps, 5 frames = 167 ms;
# at 60 fps, 5 frames = 83 ms. A fixed time value gives consistent feel regardless
# of camera speed. 0.25 s means the key can fire at most 4 times per second,
# and a single closed eye never re-fires while still shut.
WINK_DEBOUNCE_SEC = 0.25  # seconds to lock out after a wink/blink fires

# ── Flappy Bird — strict flap debounce ──────────────────────────
# Flappy Bird is instant-death on a misfire.  This is double the normal
# wink debounce (0.25 s) so one eyebrow raise can never register twice.
# Raise this value if a slow raise still double-fires; lower it only
# if the control starts to feel laggy during rapid tapping sequences.
FLAPPY_BIRD_DEBOUNCE_SEC = 0.5

# ── Space Invaders — shoot repeat interval ────────────────────────
# While the eyebrow is held, the shoot key fires every this many ms.
# 150 ms ≈ 6-7 shots/second — rapid fire without fatigue.
# Lower = faster; raise if accidental double-shots occur.
SHOOT_REPEAT_INTERVAL_MS = 150

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
keys = {
    # ── ATTACK MODE (Donkey Kong, Snake, OG Snake) ─────────────────
    # All three games share the same key layout: arrows to move,
    # X = primary action, C = secondary, Z = hold/shield.
    "Attack": {
        "Blink":     'x',   # DK: jump tap | Snake: dash burst
        "LeftWink":  'c',   # Snake/OG: bonus food spawn
        "RightWink": 'v',   # unmapped — available for future use
        "Eyebrow":   'z',   # DK: hold jump | Snake: hold shield/slow-mo
    },

    # ── PACMAN MODE ────────────────────────────────────────────────
    # Head arrows = direction. Eyebrow hold = slow ghosts (Z key).
    # Blink = restart after game over (X key).
    "Pacman": {
        "Blink":    'x',   # restart on game-over screen
        "Eyebrow":  'z',   # hold to slow all ghosts
    },

    # ── INTERACT MODE (WASD games) ─────────────────────────────────
    "Interact": {
        "Blink":     'e',   # full blink → interact / confirm
        "LeftWink":  'c',   # left wink  → dash
        "RightWink": 'v',   # right wink → special
        "Eyebrow":   'i',   # eyebrow raise → inventory (single press)
    },

    # ── MARIO MODE (Super Mario Python) ───────────────────────────
    # LEFT/RIGHT arrows = run left/right.
    # Eyebrow hold = SPACE (jump — variable height via hold duration).
    # Blink = LEFT SHIFT (run boost for speed).
    # LeftWink = ESCAPE (pause/unpause).
    # HEAD UP fires 'up' arrow which also triggers jump — intentional
    # (useful for entering pipes and climbing vines).
    "Mario": {
        "Blink":     'shift',   # run / speed boost (left shift)
        "LeftWink":  'escape',  # pause / unpause
        "RightWink": 'return',  # menu confirm / start level
        "Eyebrow":   'space',   # jump (hold = tall jump, fast flick = short hop)
    },

    # ── FLAPPY BIRD MODE ─────────────────────────────────────────
    # One-button game: blink OR eyebrow raise → SPACE (flap).
    # Head movement is DISABLED in this mode (UP arrow also flaps
    # in the game so accidental head tilts would kill the run).
    "Flappy_Bird": {
        "Blink":   'space',   # blink → flap
        "Eyebrow": 'space',   # eyebrow raise → flap (single instant press)
    },

    # ── SNAKE MODE ───────────────────────────────────────────────
    # Head tilt steers the snake. Arrow keys only — no diagonal.
    "Snake": {
        "Blink":     'x',   # dash burst
        "LeftWink":  'c',   # bonus food spawn
        "Eyebrow":   'z',   # hold slow-mo / shield
    },

    # ── SPACE INVADERS MODE ──────────────────────────────────────
    # HEAD LEFT/RIGHT = move ship. Eyebrow hold = shoot (repeat).
    # Blink = RETURN (start game from menu / pause mid-game).
    "Space_Invaders": {
        "Blink":   'return',  # start game on menu / pause
        "Eyebrow": 'space',   # hold brows = shoot laser (repeating)
    },
}

head_keys = {
    "Attack"        : {"left": "left", "right": "right", "up": "up",  "down": "down"},
    "Snake"         : {"left": "left", "right": "right", "up": "up",  "down": "down"},
    "Pacman"        : {"left": "left", "right": "right", "up": "up",  "down": "down"},
    "Interact"      : {"left": "a",    "right": "d",      "up": "w",   "down": "s"},
    "Mario"         : {"left": "left", "right": "right", "up": "up",  "down": "down"},
    # Flappy Bird: head movement intentionally not used — see main loop guard.
    "Flappy_Bird"   : {"left": "left", "right": "right", "up": "up",  "down": "down"},
    "Space_Invaders": {"left": "left", "right": "right", "up": "up",  "down": "down"},
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

# ── Modes that use eyebrow HOLD (keyDown/keyUp) ──────────────────
# All other modes use eyebrow PRESS (single tap).
# Space_Invaders has its own repeat-fire logic but also uses hold state.
_BROW_HOLD_MODES = frozenset({"Attack", "Snake", "Mario", "Pacman", "Space_Invaders"})

# ── HUD label shown while brow is actively held ──────────────────
_BROW_ACTIVE_LABEL = {
    "Attack":         "JUMPING / SHIELD",
    "Snake":          "SLOW-MO",
    "Pacman":         "SLOW GHOSTS",
    "Mario":          "JUMPING",
    "Flappy_Bird":    "FLAPPING",
    "Space_Invaders": "SHOOTING",
    "Interact":       "INVENTORY",
}

# ── Direction mirror map ──────────────────────────────────────────
# The head model was trained with the camera flipped, but the flipped
# landmark coordinates cause LEFT↔RIGHT to be inverted at runtime.
# Swap them here so turning right moves the character right.
_DIR_MIRROR = {
    "LEFT":       "RIGHT",
    "RIGHT":      "LEFT",
    "UP_LEFT":    "UP_RIGHT",
    "UP_RIGHT":   "UP_LEFT",
    "DOWN_LEFT":  "DOWN_RIGHT",
    "DOWN_RIGHT": "DOWN_LEFT",
}

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
blink_hold_start = None   # wall-clock time when sustained blink began (for hold-to-exit)

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

# ── Flappy Bird flap debounce ─────────────────────────────────────
# Timestamp after which the next flap is allowed.  0.0 = ready now.
# Separate from wink debounce so FLAPPY_BIRD_DEBOUNCE_SEC can be tuned
# independently without touching any other mode.
flappy_brow_until = 0.0

# ── Space Invaders shoot repeat ───────────────────────────────────
# Timestamp for the next scheduled shoot-key press.
# Set to time.time() on the frame the eyebrow enters shoot state so
# the first shot fires immediately; subsequent shots fire every
# SHOOT_REPEAT_INTERVAL_MS milliseconds while the brow stays raised.
shoot_next_fire_at = 0.0

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
    for key in ("left", "right", "up", "down", "a", "d", "w", "s",
                "z", "x", "space", "shift", "return", "escape"):
        pyautogui.keyUp(key)


# ═══════════════════════════════════════════════════════════════════
#  GAME LAUNCHER  (shown in the FacePlay window after calibration)
# ═══════════════════════════════════════════════════════════════════

_GAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Games")

LAUNCHER_GAMES = [
    {
        "name": "Snake",        "desc": "Smooth snake",
        "mode": "Attack",       "color": (80, 200, 80),
        "cmd": [sys.executable, os.path.join(_GAMES_DIR, "snake.py")],
        "cwd": _GAMES_DIR,
    },
    {
        "name": "OG Snake",     "desc": "Retro pixel snake",
        "mode": "Attack",       "color": (100, 230, 160),
        "cmd": [sys.executable, os.path.join(_GAMES_DIR, "og-snake.py")],
        "cwd": _GAMES_DIR,
    },
    {
        "name": "Pac-Man",      "desc": "Eat dots, dodge ghosts",
        "mode": "Pacman",       "color": (0, 220, 255),
        "cmd": [sys.executable, os.path.join(_GAMES_DIR, "PacMan", "pacman.py")],
        "cwd": os.path.join(_GAMES_DIR, "PacMan"),
    },
    {
        "name": "Donkey Kong",  "desc": "Climb to save Pauline",
        "mode": "Attack",       "color": (30, 120, 255),
        "cmd": [sys.executable,
                os.path.join(_GAMES_DIR, "Basic-OOP-Donkey-Kong-in-Python", "mario_barrel.py")],
        "cwd": os.path.join(_GAMES_DIR, "Basic-OOP-Donkey-Kong-in-Python"),
    },
    {
        "name": "Flappy Bird",  "desc": "Raise brows to flap",
        "mode": "Flappy_Bird",  "color": (255, 180, 80),
        "cmd": [sys.executable, os.path.join(_GAMES_DIR, "Flappy-bird-python", "flappy.py")],
        "cwd": os.path.join(_GAMES_DIR, "Flappy-bird-python"),
    },
    {
        "name": "Space Invaders","desc": "Hold brows to shoot",
        "mode": "Space_Invaders","color": (255, 80, 200),
        "cmd": [sys.executable, os.path.join(_GAMES_DIR, "space-invaders", "spaceinvaders.py")],
        "cwd": os.path.join(_GAMES_DIR, "space-invaders"),
    },
    {
        "name": "Super Mario",  "desc": "Run, jump, collect",
        "mode": "Mario",        "color": (60, 60, 255),
        "cmd": [sys.executable, os.path.join(_GAMES_DIR, "super-mario-python", "main.py")],
        "cwd": os.path.join(_GAMES_DIR, "super-mario-python"),
    },
]


# ── Voice → game index map (checked with 'in', order matters for substrings) ──
_LAUNCHER_VOICE_MAP = [
    ("og snake",       1),
    ("retro",          1),
    ("donkey kong",    3),
    ("donkey",         3),
    ("pac man",        2),
    ("pacman",         2),
    ("pac-man",        2),
    ("flappy bird",    4),
    ("flappy",         4),
    ("space invaders", 5),
    ("invaders",       5),
    ("super mario",    6),
    ("mario",          6),
    ("snake",          0),
    ("space",          5),
]


def _run_launcher():
    """
    FacePlay in-window game launcher — runs after calibration.

    Face gestures:
      HEAD RIGHT / LEFT  — cycle to next / previous game
      BLINK (both eyes)  — launch selected game
      EYEBROW RAISE      — also launches (alternate gesture)

    Voice:
      Say a game name ("pac man", "mario", "snake", etc.) to launch directly.

    Keyboard fallback (for testing without webcam):
      LEFT / RIGHT or A / D — cycle
      SPACE or ENTER        — launch
      Q / ESC               — quit without launching

    Returns the selected game dict so the caller can auto-set ControlMode,
    or None if the user quit.
    """
    selected = 0
    n        = len(LAUNCHER_GAMES)

    # ── Voice selection thread ─────────────────────────────────────
    _voice_q: queue.Queue = queue.Queue()

    def _launcher_voice():
        lsr = sr.Recognizer()
        lsr.energy_threshold         = 300
        lsr.pause_threshold          = 0.5
        lsr.dynamic_energy_threshold = False
        try:
            with sr.Microphone() as src:
                lsr.adjust_for_ambient_noise(src, duration=0.5)
        except Exception:
            return
        while _voice_q.empty():   # stop once a game is selected
            try:
                with sr.Microphone() as src:
                    try:
                        audio = lsr.listen(src, timeout=3, phrase_time_limit=3)
                    except sr.WaitTimeoutError:
                        continue
                text = lsr.recognize_google(audio).lower()
                print(f"Launcher voice: {text}")
                for phrase, idx in _LAUNCHER_VOICE_MAP:
                    if phrase in text:
                        _voice_q.put(idx)
                        break
            except sr.UnknownValueError:
                pass
            except Exception as e:
                print(f"Launcher voice error: {e}")
                time.sleep(0.5)

    threading.Thread(target=_launcher_voice, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame   = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]
        now     = time.time()

        # ── Launch on voice command ────────────────────────────────
        if not _voice_q.empty():
            idx = _voice_q.get()
            g   = LAUNCHER_GAMES[idx]
            g["proc"] = subprocess.Popen(g["cmd"], cwd=g["cwd"])
            print(f"FacePlay Launcher (voice): launching '{g['name']}' → mode {g['mode']}")
            return g

        # ── Draw HUD ───────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, img_h), (8, 8, 18), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        # Title
        cv2.putText(frame, "FacePlay",
                    (img_w // 2 - 108, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(frame, "Say the game name to launch   e.g. \"pac man\", \"mario\", \"snake\"",
                    (img_w // 2 - 275, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 175), 1)

        # Cards
        card_w, card_h = 168, 152
        gap  = 186
        cx   = img_w // 2
        cy   = img_h // 2

        for offset in range(-2, 3):
            idx = selected + offset
            if not (0 <= idx < n):
                continue
            g      = LAUNCHER_GAMES[idx]
            color  = g["color"]          # BGR tuple
            is_sel = (idx == selected)
            sx     = cx + offset * gap - card_w // 2
            sy     = cy - card_h // 2 - (18 if is_sel else 0)

            # Card background
            bg = frame.copy()
            cv2.rectangle(bg, (sx, sy), (sx + card_w, sy + card_h), (22, 22, 38), -1)
            cv2.addWeighted(bg, 0.75 if is_sel else 0.45, frame,
                            0.25 if is_sel else 0.55, 0, frame)

            # Border
            cv2.rectangle(frame, (sx, sy), (sx + card_w, sy + card_h),
                          color if is_sel else (65, 65, 95),
                          3 if is_sel else 1)

            # Game name
            ts  = 0.56 if is_sel else 0.45
            tc  = color if is_sel else (130, 130, 130)
            tw  = cv2.getTextSize(g["name"], cv2.FONT_HERSHEY_SIMPLEX, ts, 2)[0][0]
            cv2.putText(frame, g["name"],
                        (sx + card_w // 2 - tw // 2, sy + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, ts, tc, 2 if is_sel else 1)

            # Description
            dw = cv2.getTextSize(g["desc"], cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0][0]
            cv2.putText(frame, g["desc"],
                        (sx + card_w // 2 - dw // 2, sy + 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (185, 185, 185) if is_sel else (85, 85, 85), 1)

            # FacePlay mode badge
            mw = cv2.getTextSize(g["mode"], cv2.FONT_HERSHEY_SIMPLEX, 0.31, 1)[0][0]
            cv2.putText(frame, g["mode"],
                        (sx + card_w // 2 - mw // 2, sy + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.31,
                        color if is_sel else (75, 75, 75), 1)

            if is_sel:
                lbl = "[ SAY TO LAUNCH ]"
                lw  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0][0]
                cv2.putText(frame, lbl,
                            (sx + card_w // 2 - lw // 2, sy + card_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)

        # Navigation arrows
        if selected > 0:
            pts = np.array([[28, cy], [52, cy - 16], [52, cy + 16]], np.int32)
            cv2.fillPoly(frame, [pts], (180, 180, 210))
        if selected < n - 1:
            pts = np.array([[img_w - 28, cy], [img_w - 52, cy - 16],
                            [img_w - 52, cy + 16]], np.int32)
            cv2.fillPoly(frame, [pts], (180, 180, 210))

        # Counter
        ctr = f"{selected + 1} / {n}"
        ctw = cv2.getTextSize(ctr, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(frame, ctr, (img_w // 2 - ctw // 2, img_h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 155), 1)

        cv2.imshow("FacePlay", frame)

        # ── Keyboard fallback (Q / ESC to quit) ───────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            return None


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

    key_phrases = ["pause", "resume", "fight", "interact", "mario", "flappy", "space", "invaders", "select"]
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
                    _mode = ControlMode
                # Space Invaders: ESC quits the game — skip; Flappy Bird: no pause
                if _mode not in ("Space_Invaders", "Flappy_Bird"):
                    pyautogui.press('escape')
                last_said = "pause"

            elif "resume" in text and last_said != "resume":
                with state_lock:
                    is_paused = False
                    _mode = ControlMode
                if _mode not in ("Space_Invaders", "Flappy_Bird"):
                    pyautogui.press('escape')
                last_said = "resume"

            elif "select" in text:
                _post_key_partial("super mario", _VK_RETURN)
                print("COMMAND: select → Enter")

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

            elif "mario" in text:
                with state_lock:
                    ControlMode = "Mario"
                release_all_keys()
                print("COMMAND: switched to Mario mode")

            elif "flappy" in text:
                with state_lock:
                    ControlMode = "Flappy_Bird"
                release_all_keys()
                print("COMMAND: switched to Flappy Bird mode")

            elif "space" in text or "invaders" in text:
                with state_lock:
                    ControlMode = "Space_Invaders"
                release_all_keys()
                print("COMMAND: switched to Space Invaders mode")

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

# ═══════════════════════════════════════════════════════════════════
#  GUIDED 5-STEP CALIBRATION
#  Step 1 — Neutral (2 s):      baseline EAR + brow ratio
#  Step 2 — Blink both (1.5 s): both eyes closed → full-blink threshold
#  Step 3 — Left wink (1.5 s):  only left eye closed → left-wink threshold
#  Step 4 — Right wink (1.5 s): only right eye closed → right-wink threshold
#  Step 5 — Raise brows (1.5 s): brows up → brow-raise threshold
#
#  Each threshold is set to the midpoint between the neutral value and the
#  minimum observed during that gesture step, so it is personalised to the
#  user's own facial geometry.
# ═══════════════════════════════════════════════════════════════════

def _run_calib_step(cap, detector, duration, title, subtitle, color):
    """Collect face data for `duration` seconds while showing the given HUD."""
    l_ears, r_ears, brows = [], [], []
    step_start = time.time()
    while time.time() - step_start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        try:
            ts = int(time.time() * 1000)
            res = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            if res.face_landmarks:
                face = res.face_landmarks[0]
                l_ears.append(EAR(face, LEFT_EAR_IDS))
                r_ears.append(EAR(face, RIGHT_EAR_IDS))
                brows.append(get_brow_raise(face))
        except Exception:
            pass
        remaining = max(0.0, duration - (time.time() - step_start))
        overlay = frame.copy()
        cv2.rectangle(overlay, (40, 110), (600, 380), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, title,
                    (60, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, subtitle,
                    (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1)
        cv2.putText(frame, f"{remaining:.1f}s",
                    (280, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.imshow("FacePlay", frame)
        cv2.waitKey(1)
    return l_ears, r_ears, brows


def _show_ready(cap, message, duration=1.0):
    """Show a brief confirmation screen between calibration steps."""
    t0 = time.time()
    while time.time() - t0 < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 150), (590, 330), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, message,
                    (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 120), 2)
        cv2.imshow("FacePlay", frame)
        cv2.waitKey(1)


print("FacePlay: starting guided calibration.")

# ── Step 1: Neutral ───────────────────────────────────────────────
_show_ready(cap, "STEP 1/5 — Look straight ahead", duration=1.2)
l_n, r_n, b_n = _run_calib_step(
    cap, detector, CALIB_DURATION,
    "NEUTRAL — look straight ahead",
    "Relax your face naturally",
    (255, 255, 255))

neutral_l_ear = sum(l_n) / len(l_n) if l_n else 0.20
neutral_r_ear = sum(r_n) / len(r_n) if r_n else 0.20
neutral_ear   = (neutral_l_ear + neutral_r_ear) / 2.0
neutral_brow  = sum(b_n) / len(b_n) if b_n else 0.35

# ── Step 2: Full blink ────────────────────────────────────────────
_show_ready(cap, "STEP 2/5 — Get ready to BLINK", duration=1.0)
l_blink, r_blink, _ = _run_calib_step(
    cap, detector, 1.5,
    "BLINK — close BOTH eyes",
    "Blink repeatedly with both eyes",
    (0, 200, 255))

blink_l_min = min(l_blink) if l_blink else 0.0
blink_r_min = min(r_blink) if r_blink else 0.0
WINK_THRESHOLD = (neutral_ear + (blink_l_min + blink_r_min) / 2.0) / 2.0

# ── Step 3: Left wink ─────────────────────────────────────────────
_show_ready(cap, "STEP 3/5 — Get ready to wink LEFT", duration=1.0)
l_lwink, r_lwink, _ = _run_calib_step(
    cap, detector, 1.5,
    "LEFT WINK — close LEFT eye only",
    "Wink your left eye repeatedly",
    (80, 255, 80))

lwink_l_min = min(l_lwink) if l_lwink else 0.0
LEFT_WINK_THRESHOLD = (neutral_l_ear + lwink_l_min) / 2.0

# ── Step 4: Right wink ───────────────────────────────────────────
_show_ready(cap, "STEP 4/5 — Get ready to wink RIGHT", duration=1.0)
l_rwink, r_rwink, _ = _run_calib_step(
    cap, detector, 1.5,
    "RIGHT WINK — close RIGHT eye only",
    "Wink your right eye repeatedly",
    (80, 180, 255))

rwink_r_min = min(r_rwink) if r_rwink else 0.0
RIGHT_WINK_THRESHOLD = (neutral_r_ear + rwink_r_min) / 2.0

# ── Step 5: Eyebrow raise ─────────────────────────────────────────
_show_ready(cap, "STEP 5/5 — Get ready to raise EYEBROWS", duration=1.0)
_, _, b_raise = _run_calib_step(
    cap, detector, 1.5,
    "EYEBROWS — raise both brows",
    "Raise your eyebrows as high as you can",
    (255, 180, 0))

brow_max = max(b_raise) if b_raise else neutral_brow * 1.4
# Trigger = midpoint between neutral and the peak raise observed.
# Release = slightly below trigger for comfortable hysteresis.
BROW_TRIGGER_THRESHOLD = (neutral_brow + brow_max) / 2.0
BROW_RELEASE_THRESHOLD = neutral_brow * BROW_RELEASE_MULTIPLIER
BROW_TRIGGER_TURNED    = BROW_TRIGGER_THRESHOLD + 0.07
BROW_RELEASE_TURNED    = BROW_TRIGGER_TURNED - 0.035

# ── Print summary ─────────────────────────────────────────────────
print(f"Calibration done:"
      f"\n  neutral EAR   L={neutral_l_ear:.3f}  R={neutral_r_ear:.3f}"
      f"\n  blink thr     {WINK_THRESHOLD:.3f}"
      f"\n  left-wink thr {LEFT_WINK_THRESHOLD:.3f}"
      f"\n  right-wink thr{RIGHT_WINK_THRESHOLD:.3f}"
      f"\n  brow trigger  {BROW_TRIGGER_THRESHOLD:.3f}  release {BROW_RELEASE_THRESHOLD:.3f}")

# ── Confirmation ──────────────────────────────────────────────────
_show_ready(cap, "Calibrated!  Starting FacePlay...", duration=1.5)

# ── Load ML head-direction model ─────────────────────────────────
try:
    head_model.load()
except FileNotFoundError as e:
    print(e)
    print("FacePlay: no head_model.pkl — head movement will be disabled.")

# ── Game launcher ─────────────────────────────────────────────────
# When launched by launcher.py via --skip-launcher --mode <mode>,
# skip the in-built OpenCV launcher and go straight to face control.
_ext_mode = None
if "--skip-launcher" in sys.argv:
    try:
        _ext_mode = sys.argv[sys.argv.index("--mode") + 1]
    except (ValueError, IndexError):
        _ext_mode = "Attack"

if _ext_mode:
    _launched_game = {"mode": _ext_mode, "name": _ext_mode, "proc": None}
else:
    _launched_game = _run_launcher()
    if _launched_game is None:
        # User quit from the launcher — exit cleanly
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Auto-switch ControlMode to match the launched game
with state_lock:
    ControlMode = _launched_game["mode"]
prev_mode = ControlMode
print(f"FacePlay: entering main loop in {ControlMode} mode — press Q to quit.")

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
        flappy_brow_until  = 0.0   # reset Flappy Bird debounce on mode change
        shoot_next_fire_at = 0.0   # reset Space Invaders shoot timer on mode change
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
        leftOnly   = leftValue  < LEFT_WINK_THRESHOLD  and rightValue >= WINK_THRESHOLD
        rightOnly  = rightValue < RIGHT_WINK_THRESHOLD and leftValue  >= WINK_THRESHOLD

        # ── Hold-blink exit (3 s sustained blink → close game) ───
        # Only active in Pacman mode. Tracks wall-clock time of continuous
        # eye closure; terminates the game subprocess after 3 seconds.
        if current_mode == "Pacman":
            if bothClosed:
                if blink_hold_start is None:
                    blink_hold_start = time.time()
                elif time.time() - blink_hold_start >= 3.0:
                    proc = _launched_game.get("proc")
                    if proc and proc.poll() is None:
                        proc.terminate()
                    blink_hold_start = None
            else:
                blink_hold_start = None
        else:
            blink_hold_start = None

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
                if current_mode == "Flappy_Bird":
                    _post_key_to_window("Flappy Bird", _VK_SPACE)
                else:
                    pyautogui.press(res)
            if current_mode == "Attack":
                action_text = "ATTACKING"
            elif current_mode == "Mario":
                action_text = "RUN BOOST"
            elif current_mode == "Space_Invaders":
                action_text = "PAUSE/START"
            else:
                action_text = "INTERACTING"
            action_text_until = now + ACTION_FEEDBACK_DUR

        else:
            # ── Left wink only ────────────────────────────────────
            if leftOnly and now >= leftWinkUntil:
                leftWinkUntil = now + WINK_DEBOUNCE_SEC
                res = keys.get(current_mode, {}).get("LeftWink")
                if res:
                    pyautogui.press(res)
                action_text       = "LEFT WINK"
                action_text_until = now + ACTION_FEEDBACK_DUR

            # ── Right wink only ───────────────────────────────────
            elif rightOnly and now >= rightWinkUntil:
                rightWinkUntil = now + WINK_DEBOUNCE_SEC
                res = keys.get(current_mode, {}).get("RightWink")
                if res:
                    pyautogui.press(res)
                action_text       = "RIGHT WINK"
                action_text_until = now + ACTION_FEEDBACK_DUR

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
            if current_mode in _BROW_HOLD_MODES and res:
                pyautogui.keyUp(res)
            last_brow_state = False

        if brow_up:
            brow_sustain_count += 1

            if not last_brow_state and current_mode in _BROW_HOLD_MODES:
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

            elif not last_brow_state and current_mode == "Flappy_Bird":
                # ── Flappy Bird: instant single press with strict debounce ──
                # No sustain wait — Flappy Bird demands zero-latency response.
                # FLAPPY_BIRD_DEBOUNCE_SEC (0.5 s) stops one eyebrow raise from
                # registering twice and killing the run.  Uses its own debounce
                # so the window is tunable independently of the wink debounce.
                if now >= flappy_brow_until:
                    flappy_brow_until = now + FLAPPY_BIRD_DEBOUNCE_SEC
                    last_brow_state   = True
                    if res:
                        _post_key_to_window("Flappy Bird", _VK_SPACE)

            elif not last_brow_state and current_mode == "Space_Invaders":
                # ── Space Invaders: enter shoot-hold state ───────────────
                # The first shot fires on this same frame (shoot_next_fire_at
                # is set to now). Subsequent shots repeat every
                # SHOOT_REPEAT_INTERVAL_MS while the eyebrow stays raised —
                # handled in the repeat block below, outside this entry guard.
                if brow_sustain_count >= BROW_SUSTAIN_FRAMES:
                    last_brow_state    = True
                    shoot_next_fire_at = now  # fire immediately on entry

            elif not last_brow_state and current_mode == "Interact":
                # Interact mode: single press after sustain (unchanged)
                if brow_sustain_count >= BROW_SUSTAIN_FRAMES:
                    last_brow_state = True
                    if res:
                        pyautogui.press(res)

        # ── Space Invaders: repeat shoot while eyebrow is held ────────
        # Runs every frame while in shoot-hold state, independent of the
        # brow_up entry logic above.  shoot_next_fire_at was set to now
        # on the entry frame so the first shot fires immediately; all
        # subsequent shots fire on the timer without re-checking brow_up.
        if current_mode == "Space_Invaders" and last_brow_state:
            if res and now >= shoot_next_fire_at:
                pyautogui.press(res)
                shoot_next_fire_at = now + SHOOT_REPEAT_INTERVAL_MS / 1000.0

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
                if current_mode in _BROW_HOLD_MODES and res:
                    pyautogui.keyUp(res)
                if current_mode == "Space_Invaders":
                    shoot_next_fire_at = 0.0  # stop repeating when brow drops
                # Flappy_Bird: last_brow_state resets here; no keyUp needed
                # (was a press, not a hold) — next raise is now allowed.

        if last_brow_state:
            active_gesture = _BROW_ACTIVE_LABEL.get(current_mode, "BROW ACTIVE")

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
            flappy_brow_until    = 0.0   # reset Flappy Bird debounce
            shoot_next_fire_at   = 0.0   # cancel Space Invaders shoot repeat
            horiz_active         = None
            last_direction     = None
            release_all_keys()

        seconds_missing = time.time() - face_lost_time

        # Auto-pause after NO_FACE_PAUSE_DELAY seconds — press ESC once only
        # Skip for Space Invaders (ESC quits) and Flappy Bird (no pause)
        if seconds_missing >= NO_FACE_PAUSE_DELAY and not auto_paused:
            if current_mode not in ("Space_Invaders", "Flappy_Bird"):
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

    # Flappy Bird has no directional movement — UP arrow also flaps in that game,
    # so we disable all head-direction key sending for this mode.
    _head_dir_active = (current_mode != "Flappy_Bird")

    if face_detected and _head_dir_active:
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
        # Mario mode: hold both keys simultaneously (jump + run).
        # All other modes: fire two quick successive presses (combo input).
        # One-shot: fires only when first entering the diagonal direction.
        # horiz_active is cleared so the horizontal hysteresis doesn't fight.
        if direction in _DIAGONALS:
            if horiz_active is not None:
                horiz_active = None   # diagonal takes over; clear latched horiz
            if direction != last_direction:
                if current_mode == "Mario":
                    # Release all directional keys first (handles diagonal→diagonal
                    # transitions cleanly), then hold both simultaneously.
                    _km = head_keys["Mario"]
                    pyautogui.keyUp(_km["left"]);  pyautogui.keyUp(_km["right"])
                    pyautogui.keyUp(_km["up"]);    pyautogui.keyUp(_km["down"])
                    if direction == "UP_RIGHT":
                        pyautogui.keyDown(_km["up"]);   pyautogui.keyDown(_km["right"])
                    elif direction == "UP_LEFT":
                        pyautogui.keyDown(_km["up"]);   pyautogui.keyDown(_km["left"])
                    elif direction == "DOWN_RIGHT":
                        pyautogui.keyDown(_km["down"]); pyautogui.keyDown(_km["right"])
                    elif direction == "DOWN_LEFT":
                        pyautogui.keyDown(_km["down"]); pyautogui.keyDown(_km["left"])
                else:
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
                active_gesture = "JUMP + LEFT" if current_mode == "Mario" else "FLICK UP → LEFT"
            elif last_direction == "UP_RIGHT":
                active_gesture = "JUMP + RIGHT" if current_mode == "Mario" else "FLICK UP → RIGHT"
            elif last_direction == "DOWN_LEFT":
                active_gesture = "DOWN + LEFT" if current_mode == "Mario" else "FLICK DOWN → LEFT"
            elif last_direction == "DOWN_RIGHT":
                active_gesture = "DOWN + RIGHT" if current_mode == "Mario" else "FLICK DOWN → RIGHT"
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
    # Orange for Attack, blue for Interact, red for Mario
    if current_mode == "Attack":
        mode_label, mode_color = "ATTACK",          (0, 120, 255)
    elif current_mode == "Pacman":
        mode_label, mode_color = "PAC-MAN",         (0, 220, 255)
    elif current_mode == "Interact":
        mode_label, mode_color = "INTERACT",        (200, 140, 0)
    elif current_mode == "Mario":
        mode_label, mode_color = "MARIO",           (0, 60, 220)
    elif current_mode == "Flappy_Bird":
        mode_label, mode_color = "FLAPPY BIRD",     (0, 210, 255)
    elif current_mode == "Space_Invaders":
        mode_label, mode_color = "SPACE INVADERS",  (0, 220, 80)
    else:
        mode_label, mode_color = current_mode.upper(), (255, 255, 255)
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
