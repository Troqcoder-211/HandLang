import numpy as np
import glob
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf


# hàm pad sequences
def pad_sequences_list(X_list, maxlen):
    n_feat = X_list[0].shape[1]
    Xp = np.zeros((len(X_list), maxlen, n_feat), dtype=np.float32)
    for i, x in enumerate(X_list):
        L = x.shape[0]
        Xp[i, :L, :] = x
    return Xp


files = glob.glob("data/keypoints/*/*.npz")
X_list, y_list, labels = [], [], []
label_map = {}
for f in files:
    lbl = os.path.basename(os.path.dirname(f))
    if lbl not in label_map:
        label_map[lbl] = len(label_map)
    arr = np.load(f)["arr_0"]  # shape (frames, 63)
    X_list.append(arr.astype("float32"))
    y_list.append(label_map[lbl])

maxlen = max(x.shape[0] for x in X_list)
n_feat = X_list[0].shape[1]
# pad to maxlen
X = np.zeros((len(X_list), maxlen, n_feat), dtype=np.float32)
for i, x in enumerate(X_list):
    X[i, : x.shape[0], :] = x
y = to_categorical(y_list, num_classes=len(label_map))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y_list)

inputs = tf.keras.Input(shape=(maxlen, n_feat))
x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
x = tf.keras.layers.LSTM(64)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
out = tf.keras.layers.Dense(len(label_map), activation="softmax")(x)
model = tf.keras.Model(inputs, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint("models/seq_best.h5", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    ],
)
