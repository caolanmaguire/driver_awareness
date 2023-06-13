import cv2
import time
import mediapipe as mp

def hand_pose_prediction(mp_drawing,mp_holistic,holistic_model,capture):
    
    while capture.isOpened():
        # capture frame by frame
        ret, frame = capture.read()

        # resizing the frame for better view
        frame = cv2.resize(frame, (800, 600))

        # Converting the from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Making predictions using holistic model
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True

        # Converting back the RGB image to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Drawing the Facial Landmarks
        # mp_drawing.draw_landmarks(
        # image,
        # results.face_landmarks,
        # mp_holistic.FACEMESH_CONTOURS,
        # # mp_drawing.DrawingSpec(
        # # 	color=(255,0,255),
        # # 	thickness=1,
        # # 	circle_radius=1
        # # ),
        # mp_drawing.DrawingSpec(
        # 	color=(0,255,255),
        # 	thickness=1,
        # 	circle_radius=1
        # )
        # )

        # Drawing Right hand Land Marks
        mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
        )

        # Drawing Left hand Land Marks
        mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
        )
        
        # # Calculating the FPS
        # currentTime = time.time()
        # fps = 1 / (currentTime-previousTime)
        # previousTime = currentTime
        
        # # Displaying FPS on the image
        # cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

        # Display the resulting image
        cv2.imshow("Facial and Hand Landmarks", image)

        # Enter key 'q' to break the loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break