import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA_FILE = "hand_sign_data.csv"
CLEAN_FILE = "hand_sign_data_clean.csv"
MODEL_FILE = "hand_sign_model.pkl"

# Chọn file dữ liệu
file_to_use = CLEAN_FILE if os.path.exists(CLEAN_FILE) else DATA_FILE
print(f"📂 Đang đọc dữ liệu từ: {file_to_use}")

# Đọc dữ liệu
data = pd.read_csv(file_to_use)

# Loại bỏ nhãn rác nếu còn sót
data = data[~data["label"].isin(["label", "\r"])]

if "label" not in data.columns:
    raise ValueError("❌ Không tìm thấy cột 'label' trong dữ liệu!")

# Tách features và labels
X = data.drop("label", axis=1)
y = data["label"]

# Encode nhãn
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train model
print("🚀 Đang train model...")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Đánh giá
y_pred = clf.predict(X_test)
print("\n📊 Kết quả đánh giá:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# Lưu model + label encoder
joblib.dump((clf, le), MODEL_FILE)
print(f"✅ Model đã được lưu tại {MODEL_FILE}")
