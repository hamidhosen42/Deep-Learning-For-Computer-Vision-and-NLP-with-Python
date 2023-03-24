import mediapipe as mp
import cv2

handModel=mp.solutions.hands
handModelDrawing=mp.solutions.drawing_utils
webcam=cv2.VideoCapture(0)

with handModel.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
    while True:
        control,frame=webcam.read()
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=hands.process(rgb)
        if result.multi_hand_landmarks:
            for handLandmark in result.multi_hand_landmarks:
                handModelDrawing.draw_landmarks(frame,handLandmark,handModel.HAND_CONNECTIONS)
        cv2.imshow("Hands",frame)
        if cv2.waitKey(10)==27:
            break