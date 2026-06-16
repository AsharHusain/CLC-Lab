# =========================================================
# IMPORTS
# =========================================================

import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input,
    Dense,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "dwpose_keypoints.csv")
IMAGE_ROOT = os.path.join(BASE_DIR, "Uniform_Skeletal_Images")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

# =========================================================
# LOGGING
# =========================================================

RESULT_LOG_PATH = os.path.join(BASE_DIR, "results_log.txt")

class TeeLogger:

    def __init__(self, filepath):

        self.terminal = sys.stdout

        self.log = open(
            filepath,
            "a",
            encoding="utf-8",
            buffering=1
        )

    def write(self, message):

        self.terminal.write(message)

        self.log.write(message)

        self.log.flush()

    def flush(self):

        self.terminal.flush()

        self.log.flush()


sys.stdout = TeeLogger(RESULT_LOG_PATH)
sys.stderr = sys.stdout

# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv(CSV_PATH)

print("\nTotal frames:", len(df))

# =========================================================
# UTILITIES
# =========================================================

def normalize_path(path):

    return str(path).replace("\\", "/").rstrip("/")


def extract_video_name(path):

    path = normalize_path(path)

    return os.path.basename(
        os.path.dirname(path)
    )


def extract_word_name(path):

    path = normalize_path(path)

    return os.path.basename(
        os.path.dirname(
            os.path.dirname(path)
        )
    )


def extract_video_number(video_name):

    try:

        parts = str(video_name).split("_")

        if len(parts) < 2:
            return -1

        return int(parts[-1])

    except:

        print(f"[WARN] Invalid video folder: {video_name}")

        return -1

# =========================================================
# PATH PARSING
# =========================================================

df["video_name"] = df["skeletal_image_path"].apply(
    extract_video_name
)

df["word_name"] = df["skeletal_image_path"].apply(
    extract_word_name
)

df["video_number"] = df["video_name"].apply(
    extract_video_number
)

# remove invalid rows
df = df[df["video_number"] != -1].copy()

print("\nRemaining frames:", len(df))

# =========================================================
# BUILD IMAGE PATHS
# =========================================================

def build_image_path(row):

    original_path = normalize_path(
        row["skeletal_image_path"]
    )

    image_name = os.path.basename(
        original_path
    )

    video_folder = row["video_name"]

    word_name = row["word_name"]

    new_path = os.path.join(
        IMAGE_ROOT,
        word_name,
        video_folder,
        image_name
    )

    return new_path

df["new_image_path"] = df.apply(
    build_image_path,
    axis=1
)

print("\nExample image path:")
print(df["new_image_path"].iloc[0])

# =========================================================
# LABEL ENCODING
# =========================================================

le = LabelEncoder()

y = le.fit_transform(
    df["label"].values
)

num_classes = len(le.classes_)

y_cat = to_categorical(
    y,
    num_classes
)

# =========================================================
# FIXED VIDEO SPLIT
# =========================================================

video_nums = df["video_number"].values

train_mask = np.isin(
    video_nums,
    [1,2,3,4,5,6,7]
)

val_mask = np.isin(
    video_nums,
    [8]
)

test_mask = np.isin(
    video_nums,
    [9,10]
)

# =========================================================
# TRAIN / TEST DATAFRAMES
# =========================================================

train_df = df[train_mask].reset_index(drop=True)
val_df = df[val_mask].reset_index(drop=True)
test_df = df[test_mask].reset_index(drop=True)

y_train = y_cat[train_mask]
y_val = y_cat[val_mask]
y_test = y_cat[test_mask]

y_val_labels = y[val_mask]
y_test_labels = y[test_mask]

print("\nTrain samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))

# =========================================================
# TF DATA PIPELINE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

def parse_function(img_path, label):

    img = tf.io.read_file(img_path)

    img = tf.image.decode_jpeg(
        img,
        channels=3
    )

    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE]
    )

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    return img, label

# =========================================================
# TRAIN DATASET
# =========================================================

train_dataset = tf.data.Dataset.from_tensor_slices((
    train_df["new_image_path"].values,
    y_train.astype(np.float32)
))

train_dataset = (
    train_dataset
    .map(parse_function, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# =========================================================
# VALIDATION DATASET
# =========================================================

val_dataset = tf.data.Dataset.from_tensor_slices((
    val_df["new_image_path"].values,
    y_val.astype(np.float32)
))

val_dataset = (
    val_dataset
    .map(parse_function, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# =========================================================
# TEST DATASET
# =========================================================

test_dataset = tf.data.Dataset.from_tensor_slices((
    test_df["new_image_path"].values,
    y_test.astype(np.float32)
))

test_dataset = (
    test_dataset
    .map(parse_function, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# =========================================================
# MOBILENETV2 MODEL
# =========================================================

cnn_input = Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=cnn_input
)

# =========================================================
# FREEZE BACKBONE
# =========================================================

base_model.trainable = False

# =========================================================
# CLASSIFICATION HEAD
# =========================================================

x = GlobalAveragePooling2D()(
    base_model.output
)

output = Dense(
    num_classes,
    activation='softmax'
)(x)

# =========================================================
# FINAL MODEL
# =========================================================

model = Model(
    inputs=cnn_input,
    outputs=output
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================================
# SAVE BEST MODEL
# =========================================================

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best_mobilenet.keras")

checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

# =========================================================
# TRAIN
# =========================================================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# =========================================================
# PREDICT
# =========================================================

pred_probs = model.predict(
    test_dataset
)

preds = np.argmax(
    pred_probs,
    axis=1
)

# =========================================================
# VALIDATION ACCURACY
# =========================================================

val_probs = model.predict(val_dataset)

val_preds = np.argmax(
    val_probs,
    axis=1
)

val_acc = accuracy_score(
    y_val_labels,
    val_preds
)

print("\n" + "="*60)
print(f"VALIDATION ACCURACY: {val_acc:.4f}")
print("="*60)

# =========================================================
# TEST ACCURACY
# =========================================================

acc = accuracy_score(
    y_test_labels,
    preds
)

print("\n" + "="*60)
print(f"TEST ACCURACY: {acc:.4f}")
print("="*60)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

labels_present = np.unique(
    np.concatenate([
        y_test_labels,
        preds
    ])
)

target_names_present = le.inverse_transform(
    labels_present
)

report = classification_report(
        y_test_labels,
        preds,
        labels=labels_present,
        target_names=target_names_present,
        zero_division=0
    )

print(report)

# =========================================================
# BEST EPOCH
# =========================================================

best_epoch = np.argmax(
    history.history["val_accuracy"]
) + 1

best_acc = np.max(
    history.history["val_accuracy"]
)

print("Best Epoch:", best_epoch)
print("Best Validation Accuracy:", best_acc)