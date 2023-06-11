# IMPORTING LIBRARIES
import cv2
import mediapipe as mp
import numpy as np

# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

# DETECT THE FACE LANDMARKS
with mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh:
  while True:
    success, image = cap.read()

    # Flip the image horizontally and convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Detect the face landmarks
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert back to the BGR color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    face_3d = []
    face_2d = []
    
    # Draw the face mesh annotations on the image.
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx ==1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                #Get the 2d coordinates
                face_2d.append([x,y])

                #3d coordinates
                face_3d.append([x,y, lm.z])
        
        face_2d = np.array(face_2d, dtype=np.float64)

        face_3d = np.array(face_3d, dtype=np.float64)

        #The camera matrix
        focal_length = 1 * img_w

        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])
        
        dist_matrix = np.zeros((4,1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        rmat, jac = cv2.Rodrigues(rot_vec)

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360


        if y < -10:
            print('looking left')
        elif y > 10:
            print('looking right')
        elif x < -10:
            print('looking down')
        elif x > 10:
            print('looking up')
        else:
            print('looking forward')
        
        # print(f'results.multi_face_landmarks = ',results.multi_face_landmarks)
        # print(f'results = ',results)
        # print('\n - - - - - - - - - \n')
        
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())

    # Display the image
    cv2.imshow('Driver alertness project', image)
    
    # Terminate the process
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()