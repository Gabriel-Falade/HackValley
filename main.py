import cv2
import mediapipe as mp
import pyautogui
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

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


WINK_THRESHOLD = 12

leftEAR = [33, 159, 158, 133, 153, 145]
rightEAR = [362, 380, 374, 263, 386, 385]
leftWink = False
leftDB = 5 #in milliseconds
rightWink = False
rightDB = 5 #in milliseconds


def EAR(face, ids):
    p2p6 = abs(face[ids[1]].y - face[ids[5]].y) #p2 - p6
    p3p5 = abs(face[ids[2]].y - face[ids[4]].y) #p3 - p5
    p1p4 = abs(face[ids[0]].x - face[ids[3]].x) #p1 - p4
    return ((p2p6 + p3p5) / (2 * p1p4))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

while True:

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)

    results = detector.detect_for_video(mp_image, timestamp_ms)

    if results.face_landmarks:
        face = results.face_landmarks[0]         

        leftValue = EAR(face, leftEAR) * 100
        rightValue = EAR(face, rightEAR) * 100
    
        if leftValue < WINK_THRESHOLD and not leftWink:
            leftWink = True
            pyautogui.press('x') 
        elif leftWink:
            if leftDB > 0:
                leftDB -= 1
            else:
                leftDB = 5
                leftWink = False

        if rightValue < WINK_THRESHOLD and not rightWink:
            rightWink = True
            pyautogui.press('c') 
        elif rightWink:
            if rightDB > 0:
                rightDB -= 1
            else:
                rightDB = 5
                rightWink = False
        
        if results.face_blendshapes:
            blendshapes = results.face_blendshapes[0]
            count = 0
            for c in blendshapes:
                if c.category_name == 'browInnerUp' and c.score > 0.7:
                    count += 1
                if c.category_name == 'browOuterUpLeft' and c.score > 0.7:
                    count += 1
                if c.category_name == 'browOuterUpRight' and c.score > 0.7:
                    count += 1
            if count == 3:
                pyautogui.press('z')


        # Show face detected status
        cv2.putText(frame, "Face Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # No face found
        cv2.putText(frame, "No Face Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("FacePlay - Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ──────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
face_mesh.close()