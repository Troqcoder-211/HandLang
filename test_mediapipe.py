import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # dùng DirectShow trên Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1040)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)

if not cap.isOpened():
    print("❌ Không mở được camera")
    exit()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ Không lấy được frame từ camera")
            break
        
        # chuyển sang RGB cho Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # vẽ keypoints
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # lật ngang để màn hình hiển thị như gương
        frame = cv2.flip(frame, 1)

        cv2.imshow("MediaPipe Hands", frame)


        if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
            break

cap.release()
cv2.destroyAllWindows()
