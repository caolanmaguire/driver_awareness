import cv2
import time
import mediapipe as mp
import hand_pose_estimation

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
capture = cv2.VideoCapture(0)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

hand_pose_estimation.hand_pose_prediction(mp_drawing,mp_holistic,holistic_model,capture)


# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
