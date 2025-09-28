# capture_images.py
import cv2, os, argparse
import mediapipe as mp

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--out", default="data/images")
args = parser.parse_args()

out_dir = os.path.join(args.out, args.label)
os.makedirs(out_dir, exist_ok=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
count = len(os.listdir(out_dir))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)
    if res.multi_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
        )
    cv2.putText(
        frame,
        f"{args.label} count={count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("capture", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("s"):
        if res.multi_hand_landmarks:
            xs = [int(lm.x * w) for lm in res.multi_hand_landmarks[0].landmark]
            ys = [int(lm.y * h) for lm in res.multi_hand_landmarks[0].landmark]
            x1, x2 = max(min(xs) - 20, 0), min(max(xs) + 20, w)
            y1, y2 = max(min(ys) - 20, 0), min(max(ys) + 20, h)
            crop = frame[y1:y2, x1:x2]
            path = os.path.join(out_dir, f"{count:04d}.jpg")
            cv2.imwrite(path, crop)
        else:
            path = os.path.join(out_dir, f"{count:04d}.jpg")
            cv2.imwrite(path, frame)
        count += 1
    elif k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
