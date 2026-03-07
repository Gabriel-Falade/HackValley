import cv2 
import mediapipe as mp

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")


while True:
    # ret is return value and frame is individual fram
    ret, frame = cap.read()

    # if not succesful print error
    if not ret:
        print("Could not grab frame")
        break

    # Flip so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # MediaPipe needs RGB, OpenCV gives BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # display the camera
    cv2.imshow("FacePlay - Face Mesh", frame)

    # exit if hit q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    