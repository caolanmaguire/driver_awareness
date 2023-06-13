# IMPORTING LIBRARIES
import cv2
import mediapipe as mp
import numpy as np
import face_analysis


# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)

face_analysis.face_post_analysis(mp_face_mesh,cap,mp_drawing, mp_drawing_styles)

cap.release()
cv2.destroyAllWindows()