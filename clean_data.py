import pandas as pd

DATA_FILE = "hand_sign_data.csv"
CLEAN_FILE = "hand_sign_data_clean.csv"

df = pd.read_csv(DATA_FILE)

# Loại bỏ nhãn rác
df = df[~df["label"].isin(["label", "\r"])]

# Reset lại index
df = df.reset_index(drop=True)

# Lưu file sạch
df.to_csv(CLEAN_FILE, index=False)
print("✅ Đã làm sạch dữ liệu, lưu tại:", CLEAN_FILE)
print("📊 Phân bố mẫu mới:")
print(df["label"].value_counts())
