import cv2

for i in range(3):  # thử 0, 1, 2
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera {i} mở được")
        cap.release()
    else:
        print(f"❌ Camera {i} không mở được")
