import cv2
import mediapipe as mp
import pyautogui
import time

# ── PyAutoGUI optimization ───────────────────────────────────────
pyautogui.PAUSE = 0          # remove default 0.1s delay
pyautogui.FAILSAFE = False   # disable failsafe overhead

# ── MediaPipe setup ──────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = 'face_landmarker.task'
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    num_faces=1
)
detector = FaceLandmarker.create_from_options(options)

# ── Thresholds ───────────────────────────────────────────────────
WINK_THRESHOLD = 0.12        # lower = more sensitive
BROW_THRESHOLD = 0.7         # lower = more sensitive

# ── EAR landmark indices ─────────────────────────────────────────
LEFT_EAR_IDS  = [33, 159, 158, 133, 153, 145]
RIGHT_EAR_IDS = [362, 380, 374, 263, 386, 385]

# ── Debounce state ───────────────────────────────────────────────
leftWink  = False
rightWink = False
browRaise = False                # NEW - tracks if eyebrow raise is active
DEBOUNCE_FRAMES      = 5        # frames before another wink can trigger
BROW_DEBOUNCE_FRAMES = 10       # NEW - frames before another eyebrow raise can trigger
leftDB  = 0
rightDB = 0
browDB  = 0                      # NEW - eyebrow debounce counter

# ── Blendshape set for O(1) lookup ───────────────────────────────
BROW_SHAPES = {'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'}

# ── EAR calculation ──────────────────────────────────────────────
def EAR(face, ids):
    p2p6 = abs(face[ids[1]].y - face[ids[5]].y)
    p3p5 = abs(face[ids[2]].y - face[ids[4]].y)
    p1p4 = abs(face[ids[0]].x - face[ids[3]].x)
    return (p2p6 + p3p5) / (2.0 * p1p4)

# ── Camera setup ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Could not open camera")
    exit()

print("Running - press Q to quit")

# ── Main loop ────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    start = time.time()

    frame = cv2.flip(frame, 1)

    # Convert to RGB — mark not writeable to skip internal copy
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    timestamp_ms = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image := mp.Image(
        image_format=mp.ImageFormat.SRGB, data=rgb_frame), timestamp_ms)

    if results.face_landmarks:
        face = results.face_landmarks[0]

        # ── EAR calculation ──────────────────────────────────────
        leftValue  = EAR(face, LEFT_EAR_IDS)
        rightValue = EAR(face, RIGHT_EAR_IDS)

        # ── Left wink → dash (C) ─────────────────────────────────
        if leftValue < WINK_THRESHOLD and not leftWink:
            leftWink = True
            leftDB = DEBOUNCE_FRAMES
            pyautogui.press('c')
            cv2.putText(frame, "LEFT WINK", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        elif leftWink:
            leftDB -= 1
            if leftDB <= 0:
                leftWink = False

        # ── Right wink → attack (X) ──────────────────────────────
        if rightValue < WINK_THRESHOLD and not rightWink:
            rightWink = True
            rightDB = DEBOUNCE_FRAMES
            pyautogui.press('x')
            cv2.putText(frame, "RIGHT WINK", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        elif rightWink:
            rightDB -= 1
            if rightDB <= 0:
                rightWink = False

        # ── Eyebrow raise → jump (Z) ─────────────────────────────
        if results.face_blendshapes:
            blendshapes = results.face_blendshapes[0]
            count = sum(
                1 for shape in blendshapes
                if shape.category_name in BROW_SHAPES
                and shape.score > BROW_THRESHOLD
            )
            # NEW - debounced, only fires once per raise instead of every frame
            if count == 3 and not browRaise:
                browRaise = True
                browDB = BROW_DEBOUNCE_FRAMES
                pyautogui.press('z')
                cv2.putText(frame, "EYEBROW RAISE", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            elif browRaise:
                browDB -= 1
                if browDB <= 0:
                    browRaise = False

        # ── HUD ──────────────────────────────────────────────────
        cv2.putText(frame, "Face Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"L EAR: {leftValue:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, f"R EAR: {rightValue:.2f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    else:
        cv2.putText(frame, "No Face Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ── FPS display ──────────────────────────────────────────────
    totalTime = time.time() - start
    fps = 1 / totalTime if totalTime > 0 else 0
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("FacePlay", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()