"""Main file - multi threading implemented to run multiple camera feeds at once"""
import threading
import cv2
import mediapipe as mp
import hand_pose_estimation
import face_pose_estimation
# import logging

class cam_thread(threading.Thread):
    def __init__(self, previewName, cam_id):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.cam_id = cam_id
    def run(self):
        # print("Starting " + self.previewName)
        run_instance(self.previewName, self.cam_id)

def run_instance(to_run, cam_id):
    """instance of program, to use multi threading to run multi analysis

    Args:
        to_run (_type_): _description_
        cam_id (_type_): _description_
    """
    if to_run == 'face_pose':
        # INITIALIZING OBJECTS
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        cap = cv2.VideoCapture(cam_id)

        face_pose_estimation.face_pose_analysis(mp_face_mesh,cap,mp_drawing, mp_drawing_styles)

        cap.release()
        cv2.destroyAllWindows()
    elif to_run == 'hand_pose':
        # Grabbing the Holistic Model from Mediapipe and Initializing the Model
        mp_holistic = mp.solutions.holistic
        holistic_model = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Initializing the drawing utils for drawing the facial landmarks on image
        mp_drawing = mp.solutions.drawing_utils

        # (0) in VideoCapture is used to connect to your computer's default camera
        capture = cv2.VideoCapture(cam_id)

        hand_pose_estimation.hand_pose_prediction(mp_drawing,mp_holistic,holistic_model,capture)
        # When all the process is done
        # Release the capture and destroy all windows
        capture.release()
        cv2.destroyAllWindows()

# Create two threads as follows
# tasks = input('1 for face pose, 2 for hand pose, 1,2 for both:\n')

# logging.basicConfig(filename = 'report.log',level = \
# logging.DEBUG,format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
# logging.info('Info message')

thread1 = cam_thread("face_pose", 0)
thread2 = cam_thread("hand_pose", 1)
thread1.start()
thread2.start()
