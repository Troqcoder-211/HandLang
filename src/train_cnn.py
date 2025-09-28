# capture_keypoints.py
import cv2, os, argparse, numpy as np
import mediapipe as mp
from time import time


def normalize_landmarks(landmarks):
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # normalized [0,1]
    origin = arr[0, :2].copy()  # wrist
    arr[:, :2] -= origin
    scale = np.max(np.linalg.norm(arr[:, :2], axis=1))
    if scale > 0:
        arr[:, :2] /= scale
    return arr.flatten()  # shape 63


parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--out", default="data/keypoints")
parser.add_argument("--frames", type=int, default=30)
args = parser.parse_args()

out_dir = os.path.join(args.out, args.label)
os.makedirs(out_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
cap = cv2.VideoCapture(0)
count = len(os.listdir(out_dir))

print("Press 'r' to record one sample ({} frames), 'q' to quit".format(args.frames))
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)
    display = frame.copy()
    if res.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            display, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
        )
    cv2.imshow("record", display)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("r"):
        seq = []
        print("Recording...")
        for i in range(args.frames):
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(frame_rgb)
            if res.multi_hand_landmarks:
                vec = normalize_landmarks(res.multi_hand_landmarks[0].landmark)
            else:
                vec = np.zeros(63, dtype=np.float32)
            seq.append(vec)
            cv2.putText(
                frame,
                f"REC {i+1}/{args.frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("record", frame)
            cv2.waitKey(1)
        data = np.stack(seq, axis=0)  # (frames, 63)
        np.savez_compressed(os.path.join(out_dir, f"{count:04d}.npz"), landmarks=data)
        print("Saved sample", count)
        count += 1
    elif k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
