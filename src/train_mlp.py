import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA_FILE = "hand_sign_data.csv"
CLEAN_FILE = "hand_sign_data_clean.csv"
MODEL_FILE = "hand_sign_model_mlp.pkl"

# Chọn file dữ liệu
file_to_use = CLEAN_FILE if os.path.exists(CLEAN_FILE) else DATA_FILE
print(f"📂 Đang đọc dữ liệu từ: {file_to_use}")

# Đọc dữ liệu
data = pd.read_csv(file_to_use)

# Loại bỏ nhãn rác
data = data[~data["label"].isin(["label", "\r"])]

if "label" not in data.columns:
    raise ValueError("❌ Không tìm thấy cột 'label' trong dữ liệu!")

# Features và Labels
X = data.drop("label", axis=1)
y = data["label"]

# Encode nhãn
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale dữ liệu (quan trọng cho MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train MLP
print("🚀 Đang train MLPClassifier...")
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # 2 hidden layers
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
)
clf.fit(X_train, y_train)

# Đánh giá
y_pred = clf.predict(X_test)
print("\n📊 Kết quả đánh giá:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# Lưu model + encoder + scaler
joblib.dump((clf, le, scaler), MODEL_FILE)
print(f"✅ Model MLP đã được lưu tại {MODEL_FILE}")
