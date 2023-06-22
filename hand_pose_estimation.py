import cv2
import mediapipe as mp

def hand_pose_prediction(mp_drawing,mp_holistic,holistic_model,capture) -> None:
    
    while capture.isOpened():
        # capture frame by frame
        frame = capture.read()

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

        # Display the resulting image
        cv2.imshow("Facial and Hand Landmarks", image)

        # Enter key 'q' to break the loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
