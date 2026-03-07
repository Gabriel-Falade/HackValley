import cv2 
import mediapipe as mp
import numpy as np
import pyautogui
import time

# ── PyAutoGUI optimization ───────────────────────────────────────
pyautogui.PAUSE = 0           # remove default 0.1s delay between every pyautogui call
pyautogui.FAILSAFE = False    # disable failsafe corner detection overhead

# ── Pre-compute constants outside the loop ───────────────────────
LANDMARK_IDS = {1, 33, 61, 199, 263, 291}   # set lookup is O(1) vs list
dist_matrix = np.zeros((4, 1), dtype=np.float64)  # never changes, compute once

# ── MediaPipe setup ──────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,                  # only track 1 face, faster
    refine_landmarks=False,           # skip iris tracking, not needed here
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ── Camera setup ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # lower resolution = faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)             # request 60fps from camera

if not cap.isOpened():
    print("Could not open camera")

# ── Track last direction to avoid spamming pyautogui ────────────
last_direction = None

def get_head_direction(face_landmarks, img_w, img_h, focal_length, cam_matrix):
    face_3d = []
    face_2d = []
    
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in LANDMARK_IDS:          # O(1) set lookup
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    ret, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360

    if x > 12 and y > 10:
        return "UP_RIGHT"
    elif x > 13 and y < -13:
        return "UP_LEFT"
    elif x < -3 and y > 6:
        return "DOWN_RIGHT"
    elif x < -3 and y < -6:
        return "DOWN_LEFT"
    elif y < -10:
        return "LEFT"
    elif y > 10:
        return "RIGHT"
    elif x < -3:
        return "DOWN"
    elif x > 12:
        return "UP"
    else:
        return "FORWARD"
# ── Main loop ────────────────────────────────────────────────────
while True:
    success, frame = cap.read()
    if not success:
        print("Could not grab frame")
        break

    start = time.time()

    # Flip mirror
    frame = cv2.flip(frame, 1)

    # Get frame dimensions once
    img_h, img_w = frame.shape[:2]

    # Pre-compute cam_matrix using current frame size
    focal_length = img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]], dtype=np.float64)

    # Convert to RGB for MediaPipe - mark not writeable to skip copy
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            direction = get_head_direction(face_landmarks, img_w, img_h, focal_length, cam_matrix)

            # Only send input if direction changed — avoids spamming pyautogui every frame
            if direction != last_direction:
                if direction == "LEFT":
                    pyautogui.keyDown('a')
                    pyautogui.keyUp('d')
                    pyautogui.keyUp('w')
                    pyautogui.keyUp('s')
                elif direction == "RIGHT":
                    pyautogui.keyDown('d')
                    pyautogui.keyUp('a')
                    pyautogui.keyUp('w')
                    pyautogui.keyUp('s')
                elif direction == "UP":
                    pyautogui.keyDown('w')
                    pyautogui.keyUp('s')
                    pyautogui.keyUp('a')
                    pyautogui.keyUp('d')
                elif direction == "DOWN":
                    pyautogui.keyDown('s')
                    pyautogui.keyUp('w')
                    pyautogui.keyUp('a')
                    pyautogui.keyUp('d')
                elif direction == "UP_RIGHT":
                    pyautogui.keyDown('w')
                    pyautogui.keyDown('d')
                    pyautogui.keyUp('a')
                    pyautogui.keyUp('s')
                elif direction == "UP_LEFT":
                    pyautogui.keyDown('w')
                    pyautogui.keyDown('a')
                    pyautogui.keyUp('d')
                    pyautogui.keyUp('s')
                elif direction == "DOWN_RIGHT":
                    pyautogui.keyDown('s')
                    pyautogui.keyDown('d')
                    pyautogui.keyUp('a')
                    pyautogui.keyUp('w')
                elif direction == "DOWN_LEFT":
                    pyautogui.keyDown('s')
                    pyautogui.keyDown('a')
                    pyautogui.keyUp('d')
                    pyautogui.keyUp('w')
                elif direction == "FORWARD":
                    pyautogui.keyUp('a')
                    pyautogui.keyUp('d')
                    pyautogui.keyUp('w')
                    pyautogui.keyUp('s')
                last_direction = direction

            # HUD
            cv2.putText(frame, direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Draw face mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # FPS counter
    totalTime = time.time() - start
    fps = 1 / totalTime if totalTime > 0 else 0
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("FacePlay", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()