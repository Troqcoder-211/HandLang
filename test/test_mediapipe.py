import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # dùng DirectShow trên Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("❌ Không mở được camera")
    exit()

# Tạo 1 object từ class Hands trên mediapipe
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

    while True:
        ret, frame = cap.read()

        # ret dạng boolean => có đọc được ảnh hay không
        # frame => trả về ảnh numpy array BGR
        if not ret or frame is None:
            print("❌ Không lấy được frame từ camera")
            break

        # chuyển sang RGB cho Mediapipe
        """
        Chuyển frame từ BGR -> RGB. Quan trọng: nếu không chuyển, màu bị ngược
        và model có thể hoạt động kém.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # results sẽ trả về 1 object
        # {multi_hand_landmarks, multi_handedness, multi_hand_world_landmarks}
        results = hands.process(rgb)

        # vẽ keypoints
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # lật ngang để màn hình hiển thị như gương
        frame = cv2.flip(frame, 1)

        cv2.imshow("MediaPipe Hands", frame)

        # cv2.waitKey(1) chờ 1 ms cho phím nhấn; trả về mã phím.
        # Dùng & 0xFF để lấy 8-bit mã chuẩn. 27 = phím ESC.
        # Nếu nhấn ESC thì break.
        if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
            break

# Giải phóng camera
cap.release()
# Đóng tất cả cửa sổ OpenCV
cv2.destroyAllWindows()
