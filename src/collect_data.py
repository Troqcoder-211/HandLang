import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

CSV_FILE = "hand_sign_data.csv"

if not os.path.exists(CSV_FILE):
    pd.DataFrame().to_csv(CSV_FILE, index=False)


def extract_hand_landmarks(hand_landmarks):
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return row


def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1)

    current_label = (
        input("👉 Nhập ký hiệu cần thu thập (vd: A, B, C...): ").strip().upper()
    )
    print(f"📸 Đang thu thập dữ liệu cho ký hiệu: {current_label}")
    print("Nhấn phím [S] để lưu 1 mẫu, [Q] để thoát.")

    collected = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Vẽ thông tin hướng dẫn
                cv2.putText(
                    frame,
                    f"Label: {current_label} | Saved: {len(collected)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Lấy phím bấm ở ngoài vòng while vẽ
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):  # bấm S để lưu
                    row = extract_hand_landmarks(hand_landmarks)
                    row.append(current_label)
                    collected.append(row)
                    print(
                        f"✅ Đã lưu 1 mẫu cho {current_label} (tổng: {len(collected)})"
                    )

                elif key == ord("q"):  # bấm Q để thoát
                    cap.release()
                    cv2.destroyAllWindows()

                    if collected:
                        df = pd.DataFrame(collected)
                        if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
                            df.to_csv(CSV_FILE, mode="a", header=False, index=False)
                        else:
                            df.to_csv(CSV_FILE, index=False)
                        print(f"💾 Đã lưu {len(collected)} mẫu vào {CSV_FILE}")
                    return

        cv2.imshow("Collecting Hand Sign Data", frame)


if __name__ == "__main__":
    main()
