# test_env.py
import os
import tensorflow as tf
import cv2
import sys

# giảm log noisy từ TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # nếu muốn tắt oneDNN thông báo
# nếu bạn không có GPU hoặc muốn tắt TensorFlow tìm GPU:
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


print("Python:", sys.version.splitlines()[0])
print("TensorFlow:", tf.__version__, "GPUs:", tf.config.list_physical_devices("GPU"))
print("OpenCV:", cv2.__version__)

# thử mở camera với nhiều backend và indices
backends = [
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_MSMF", cv2.CAP_MSMF),
    ("CAP_V4L2", cv2.CAP_V4L2),
    ("CAP_GSTREAMER", cv2.CAP_GSTREAMER),
    ("CAP_ANY", cv2.CAP_ANY),
]

print("\n=== Checking /dev/video* (if exists) ===")
try:
    import glob

    devs = glob.glob("/dev/video*")
    print("devices:", devs)
except Exception:
    print("not a unix-like platform or permission denied")

print("\n=== Try open camera indices 0..3 with several backends ===")
for name, backend in backends:
    for idx in range(0, 4):
        try:
            cap = cv2.VideoCapture(idx, backend)
            opened = cap.isOpened()
            print(f"Backend {name:12} idx {idx} -> opened={opened}")
            if opened:
                ret, frame = cap.read()
                print("  read:", ret, "frame shape:", None if not ret else frame.shape)
            cap.release()
        except Exception as e:
            print(f"  Backend {name}, idx {idx} -> Exception: {e}")

print("\n=== Done ===")
