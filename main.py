import cv2
import mediapipe as mp
import csv
import os
import joblib

# Khởi tạo mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MODE = "collect"  # "collect" hoặc "predict"
DATA_FILE = "hand_sign_data.csv"
MODEL_FILE = "hand_sign_model.pkl"

clf, le = None, None
if MODE == "predict" and os.path.exists(MODEL_FILE):
    clf, le = joblib.load(MODEL_FILE)
    print("✅ Model loaded!")

current_label = None  # nhãn hiện tại
row_buffer = None  # dữ liệu keypoints của tay


def run():
    global current_label, row_buffer

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    csv_file, csv_writer = None, None

    if MODE == "collect":
        file_exists = os.path.isfile(DATA_FILE)
        csv_file = open(DATA_FILE, mode="a", newline="")
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            header = ["label"] + [
                f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]
            ]
            csv_writer.writerow(header)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Không lấy được frame từ camera")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            row_buffer = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    row = []
                    for landmark in hand_landmarks.landmark:
                        row += [landmark.x, landmark.y, landmark.z]
                    if len(row) == 63:
                        row_buffer = row

            # hiển thị label hiện tại
            if current_label:
                cv2.putText(
                    frame,
                    f"Current label: {current_label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("HandLang - Mediapipe Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and MODE == "collect":  # lưu mẫu
                if row_buffer is not None and current_label is not None:
                    csv_writer.writerow([current_label] + row_buffer)
                    print(f"💾 Lưu 1 mẫu cho nhãn: {current_label}")
            elif 32 <= key <= 126:  # ký tự ASCII (A-Z, 0-9,...)
                current_label = chr(key).upper()
                print(f"✏️ Đổi nhãn thành: {current_label}")

    cap.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
