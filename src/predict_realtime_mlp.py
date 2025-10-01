import cv2
import mediapipe as mp
import joblib

# import numpy as np
import os

MODEL_FILE = "hand_sign_model_mlp.pkl"

# Khởi tạo mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model, encoder và scaler
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("❌ Chưa có model, hãy train trước!")

clf, le, scaler = joblib.load(MODEL_FILE)
print("✅ Model MLP đã được load thành công!")


def extract_hand_landmarks(hand_landmarks):
    """Trích xuất 63 tọa độ từ Mediapipe"""
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return row


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

            label_text = "No Hand"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    row = extract_hand_landmarks(hand_landmarks)
                    if len(row) == 63:
                        # Chuẩn hóa rồi dự đoán
                        X = scaler.transform([row])
                        y_pred = clf.predict(X)
                        label = le.inverse_transform(y_pred)[0]
                        label_text = f"Sign: {label}"

            # Hiển thị nhãn dự đoán
            cv2.putText(
                frame,
                label_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )

            cv2.imshow("HandLang - Realtime Predict (MLP)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
