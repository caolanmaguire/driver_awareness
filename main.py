import cv2
import threading

to_run = 'face_pose'
# IMPORTING LIBRARIES
import cv2
import time
import mediapipe as mp
import numpy as np
import hand_pose_estimation
import face_pose_estimation

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        run_instance(self.previewName, self.camID)
        # camPreview(self.previewName, self.camID)

# def camPreview(previewName, camID):
#     cv2.namedWindow(previewName)
#     cam = cv2.VideoCapture(camID)
#     if cam.isOpened():  # try to get the first frame
#         rval, frame = cam.read()
#     else:
#         rval = False

#     while rval:
#         cv2.imshow(previewName, frame)
#         rval, frame = cam.read()
#         key = cv2.waitKey(20)
#         if key == 27:  # exit on ESC
#             break
#     cv2.destroyWindow(previewName)

def run_instance(to_run, camID):
    if to_run == 'face_pose':
        # INITIALIZING OBJECTS
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        cap = cv2.VideoCapture(camID)

        face_pose_estimation.face_post_analysis(mp_face_mesh,cap,mp_drawing, mp_drawing_styles)

        cap.release()
        cv2.destroyAllWindows()
    elif to_run == 'hand_pose':
        # Grabbing the Holistic Model from Mediapipe and
        # Initializing the Model
        mp_holistic = mp.solutions.holistic
        holistic_model = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initializing the drawing utils for drawing the facial landmarks on image
        mp_drawing = mp.solutions.drawing_utils

        # (0) in VideoCapture is used to connect to your computer's default camera
        capture = cv2.VideoCapture(camID)

        # Initializing current time and precious time for calculating the FPS
        previousTime = 0
        currentTime = 0

        hand_pose_estimation.hand_pose_prediction(mp_drawing,mp_holistic,holistic_model,capture)


        # When all the process is done
        # Release the capture and destroy all windows
        capture.release()
        cv2.destroyAllWindows()

# Create two threads as follows
thread1 = camThread("face_pose", 1)
thread2 = camThread("hand_pose", 0)
thread1.start()
thread2.start()