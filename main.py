import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while True:
    _, frame = cap.read()

    frame, faces = detector.findFaceMesh(frame)

    frame = cv2.resize(frame, (800, 450))
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)