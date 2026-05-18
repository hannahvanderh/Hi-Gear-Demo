#https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
#["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#model_path = 'hand_landmarker.task'
model_path = 'gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = vision.GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2  # Maximum number of hands to detect
)

with vision.GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = recognizer.recognize(mp_image)

        if detection_result.gestures:
            font = cv2.FONT_HERSHEY_SIMPLEX
            print(detection_result.gestures)
            for index, g in enumerate(detection_result.gestures):
                cv2.putText(frame, f"{g[0].category_name.replace("_", " ")}", (50,75 + (60 * index)), font, 2, (0, 255, 255), 5, cv2.LINE_AA)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for id, lm in enumerate(hand_landmarks):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # Custom logic for specific landmarks
                    if id == 4: # Tip of the thumb
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    else:
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # Display results
        cv2.imshow("MediaPipe Hands", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
