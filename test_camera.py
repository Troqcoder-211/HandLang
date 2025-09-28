import cv2

# mở camera mặc định (chỉ số 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được camera. Thử kiểm tra driver hoặc quyền truy cập.")
    exit()

print("✅ Camera mở thành công. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không nhận được khung hình từ camera.")
        break

    cv2.imshow("Camera", frame)

    # nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
