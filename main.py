import cv2
import mediapipe as mp

# Khởi tạo mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def run():
    cap = cv2.VideoCapture(0)  # mở camera mặc định

    # Thiết lập hand tracking
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Không lấy được frame từ camera")
                break

            frame = cv2.flip(frame, 1)

            # Chuyển sang RGB cho mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Nhận diện bàn tay
            results = hands.process(rgb)

            # Vẽ landmarks nếu có bàn tay
            if results.multi_hand_landmarks:
                row = []
                for hand_landmarks in results.multi_hand_landmarks:
                    # Vẽ điểm ảnh và nối lại trên frame
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    # Vẽ keypoints
                    for landmark in hand_landmarks.landmark:
                        row += [landmark.x, landmark.y, landmark.z]

            # Hiển thị
            cv2.imshow("HandLang - Mediapipe Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
