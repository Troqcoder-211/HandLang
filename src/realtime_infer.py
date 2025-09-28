# realtime_infer.py (tóm tắt logic)
# - load model (cnn or seq)
# - dùng mediapipe để lấy image crop (static) hoặc landmarks (sequence)
# - sequence: giữ deque(maxlen=maxlen) của vectors, khi đầy -> predict
# - smoothing: giữ history deque(preds, maxlen=10) -> mode
from collections import deque, Counter

pred_queue = deque(maxlen=10)
pred_queue.append(pred_idx)
mode = Counter(pred_queue).most_common(1)[0][0]
