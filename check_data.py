import pandas as pd
import os

DATA_FILE = "hand_sign_data.csv"

if not os.path.exists(DATA_FILE):
    print(f"❌ Không tìm thấy file {DATA_FILE}")
else:
    df = pd.read_csv(DATA_FILE)
    if "label" not in df.columns:
        print("⚠️ File CSV không có cột 'label'")
    else:
        counts = df["label"].value_counts()
        print("📊 Số lượng mẫu theo nhãn:")
        print(counts)
        print("\nTổng số mẫu:", len(df))
