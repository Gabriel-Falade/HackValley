import cv2 
import mediapipe as mp
import numpy as np
import time

# setting up drawing utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")


while True:
    # ret is return value and frame is individual fram
    success, frame = cap.read()

    start = time.time()

    # if not succesful print error
    if not success:
        print("Could not grab frame")
        break

    # Flip so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # MediaPipe needs RGB, OpenCV gives BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance
    frame.flags.writeable = False

    # Get the results 
    results = face_mesh.process(rgb_frame)

    frame.flags.writeable = True
    

    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            # convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # convert it to Numpy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array( [ [focal_length, 0, img_h / 2],
                                     [0, focal_length, img_w / 2],
                                      [0, 0, 1]])
            
            # distance matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP, may need to change the ret var
            ret, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix 
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360


            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking right"
            elif x < -10:
                text = "Looking down"
            elif x > 10:
                text = "Looking up"
            else:
                text = "Forward"


            # display the nose direction
            nose_3d_projecion, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(frame, p1, p2, (255, 0, 0), 3)

            # add the text 
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(frame, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255))
            cv2.putText(frame, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255))
            cv2.putText(frame, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255))
        
        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime if totalTime > 0 else 0

        print("FPS: ", fps)

        cv2.putText(frame, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )
    # display the camera
    cv2.imshow("FacePlay - Face Mesh", frame)

    # exit if hit q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()