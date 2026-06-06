# =========================================================
# HEADER FILES
# =========================================================
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.layers import (
    Input,
    Dense,
    Concatenate,
    GlobalAveragePooling2D,
    Add,
    ReLU,
    BatchNormalization,
    Multiply,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "dwpose_keypoints.csv")

IMAGE_ROOT = os.path.join(BASE_DIR, "Uniform_Skeletal_Images")

RESULT_LOG_PATH = os.path.join(BASE_DIR, "results_log.txt")

IMG_SIZE = 227  # AlexNet uses 227x227
BATCH_SIZE = 32
EPOCHS = 50

TOTAL_KPS = 50  # Total number of keypoints

# =========================================================
# ANGLE DEFINITIONS
# =========================================================

ANGLE_TRIPLETS = [
    # BODY
    (2, 3, 4),
    (5, 6, 7),
    (3, 2, 5),
    (6, 5, 2),
    (4, 2, 5),
    (7, 5, 2),
    # RIGHT HAND
    (8, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    (8, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    (8, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
    (8, 21, 22),
    (21, 22, 23),
    (22, 23, 24),
    (8, 25, 26),
    (25, 26, 27),
    (26, 27, 28),
    # LEFT HAND
    (29, 30, 31),
    (30, 31, 32),
    (31, 32, 33),
    (29, 34, 35),
    (34, 35, 36),
    (35, 36, 37),
    (29, 38, 39),
    (38, 39, 40),
    (39, 40, 41),
    (29, 42, 43),
    (42, 43, 44),
    (43, 44, 45),
    (29, 46, 47),
    (46, 47, 48),
    (47, 48, 49),
    # FINGER SPREAD
    (9, 8, 13),
    (13, 8, 17),
    (17, 8, 21),
    (21, 8, 25),
    (30, 29, 34),
    (34, 29, 38),
    (38, 29, 42),
    (42, 29, 46),
]

# =========================================================
# UTILITIES
# =========================================================


def normalize_path(path):

    return str(path).replace("\\", "/").rstrip("/")


def get_xy(kps, idx):

    return kps[idx * 2], kps[idx * 2 + 1]


def euclidean(x1, y1, x2, y2):

    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def angle_at_b(ax, ay, bx, by, cx, cy):

    ba = np.array([ax - bx, ay - by], dtype=np.float32)

    bc = np.array([cx - bx, cy - by], dtype=np.float32)

    n_ba = np.linalg.norm(ba)
    n_bc = np.linalg.norm(bc)

    if n_ba < 1e-6 or n_bc < 1e-6:
        return 0.0

    cos_a = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_a)))


# =========================================================
# FEATURE ENGINEERING
# =========================================================


def engineer_features(row_values):

    kps = np.array(row_values, dtype=np.float32)

    feats = list(kps)

    # =====================================================
    # ANGLES
    # =====================================================

    for a, b, c in ANGLE_TRIPLETS:

        if max(a, b, c) >= TOTAL_KPS:
            feats.append(0.0)
            continue

        ax, ay = get_xy(kps, a)
        bx, by = get_xy(kps, b)
        cx, cy = get_xy(kps, c)

        ang = angle_at_b(ax, ay, bx, by, cx, cy)

        feats.append(ang / 180.0)

    # =====================================================
    # CURL DISTANCES
    # =====================================================

    r_root = get_xy(kps, 8)
    l_root = get_xy(kps, 29)

    # right hand
    for tip_idx in [12, 16, 20, 24, 28]:

        tx, ty = get_xy(kps, tip_idx)

        dist = euclidean(tx, ty, *r_root)

        feats.append(dist)

    # left hand
    for tip_idx in [33, 37, 41, 45, 49]:

        tx, ty = get_xy(kps, tip_idx)

        dist = euclidean(tx, ty, *l_root)

        feats.append(dist)

    return np.array(feats, dtype=np.float32)


# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv(CSV_PATH)

print("\nTotal frames:", len(df))

# =========================================================
# PATH PARSING
# =========================================================


def extract_video_name(path):

    path = normalize_path(path)

    return os.path.basename(os.path.dirname(path))


def extract_word_name(path):

    path = normalize_path(path)

    return os.path.basename(os.path.dirname(os.path.dirname(path)))


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
# APPLY PATH PARSING
# =========================================================

df["video_name"] = df["skeletal_image_path"].apply(extract_video_name)

df["word_name"] = df["skeletal_image_path"].apply(extract_word_name)

df["video_number"] = df["video_name"].apply(extract_video_number)

# remove bad rows
df = df[df["video_number"] != -1].copy()

print("\nRemaining frames:", len(df))

# =========================================================
# BUILD NEW IMAGE PATHS
# =========================================================


def build_image_path(row):

    original_path = normalize_path(row["skeletal_image_path"])

    image_name = os.path.basename(original_path)

    video_folder = row["video_name"]

    word_name = row["word_name"]

    new_path = os.path.join(IMAGE_ROOT, word_name, video_folder, image_name)

    return new_path


df["new_image_path"] = df.apply(build_image_path, axis=1)

print("\nExample image path:")
print(df["new_image_path"].iloc[0])

# =========================================================
# COORD COLUMNS
# =========================================================

drop_cols = [
    "image_path",
    "skeletal_image_path",
    "new_image_path",
    "video_name",
    "video_number",
    "word_name",
    "label",
]

coord_cols = [c for c in df.columns if c not in drop_cols]

# =========================================================
# BUILD POSE FEATURES
# =========================================================

print("\nEngineering pose features...")

X_pose = np.vstack(
    [engineer_features(row) for row in df[coord_cols].replace(-1, 0).values]
)

print("Pose feature shape:", X_pose.shape)

# =========================================================
# LABEL ENCODING
# =========================================================

le = LabelEncoder()

y = le.fit_transform(df["label"].values)

num_classes = len(le.classes_)

y_cat = to_categorical(y, num_classes)

# =========================================================
# FIXED VIDEO SPLIT
# =========================================================

video_nums = df["video_number"].values

train_mask = np.isin(video_nums, [1, 2, 3, 4, 5, 6, 7, 8])

test_mask = np.isin(video_nums, [9, 10])

# =========================================================
# TRAIN / TEST DATAFRAMES
# =========================================================

train_df = df[train_mask].reset_index(drop=True)

test_df = df[test_mask].reset_index(drop=True)

# =========================================================
# POSE FEATURES
# =========================================================

X_pose_train = X_pose[train_mask]
X_pose_test = X_pose[test_mask]

# =========================================================
# LABELS
# =========================================================

y_train_labels = y[train_mask]
y_test_labels = y[test_mask]

y_train = y_cat[train_mask]
y_test = y_cat[test_mask]

print("\nTrain samples:", len(train_df))
print("Test samples:", len(test_df))

# =========================================================
# SCALE POSE FEATURES
# =========================================================

scaler = StandardScaler()

X_pose_train = scaler.fit_transform(X_pose_train)

X_pose_test = scaler.transform(X_pose_test)

# =========================================================
# CORRELATION ANALYSIS
# =========================================================

print("\nComputing correlations...")

feature_names = [f"f_{i}" for i in range(X_pose_train.shape[1])]

df_corr = pd.DataFrame(X_pose_train, columns=feature_names)

corr_matrix = df_corr.corr()

# =========================================================
# PLOT CORRELATION MATRIX
# =========================================================

plt.figure(figsize=(18, 18))

plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")

plt.colorbar()

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.show()

# =========================================================
# TOP CORRELATIONS
# =========================================================

corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)

corr_pairs = corr_pairs[corr_pairs < 1.0]

print("\nTop Correlated Features:\n")

print(corr_pairs.head(50))

# =========================================================
# REMOVE HIGHLY CORRELATED FEATURES
# =========================================================

import pandas as pd

# convert train features to dataframe
df_train_corr = pd.DataFrame(X_pose_train)

# correlation matrix
corr_matrix = df_train_corr.corr().abs()

# upper triangle only
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# =========================================================
# THRESHOLD
# =========================================================

threshold = 0.98

# =========================================================
# FIND FEATURES TO DROP
# =========================================================

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print("\n" + "=" * 60)
print("HIGHLY CORRELATED FEATURES")
print("=" * 60)

print("\nNumber of features before removal:")
print(X_pose_train.shape[1])

print("\nFeatures removed:")
print(len(to_drop))

print("\nFeature indices removed:")
print(to_drop)

# =========================================================
# REMOVE FROM TRAIN
# =========================================================

X_pose_train = df_train_corr.drop(columns=to_drop).values

# =========================================================
# REMOVE FROM TEST
# =========================================================

df_test_corr = pd.DataFrame(X_pose_test)

X_pose_test = df_test_corr.drop(columns=to_drop).values

print("\nNumber of features after removal:")
print(X_pose_train.shape[1])

print("\nNew train shape:")
print(X_pose_train.shape)

print("\nNew test shape:")
print(X_pose_test.shape)

# =========================================================
# TF DATA PIPELINE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE


def alexnet_preprocess(img):
    """Normalize image the same way AlexNet expects: zero-mean per channel."""
    img = tf.cast(img, tf.float32)
    # ImageNet channel means
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
    img = img - mean
    return img


def parse_function(img_path, pose_feat, label):

    img = tf.io.read_file(img_path)

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])  # 227x227 for AlexNet

    img = alexnet_preprocess(img)

    return ({"image_input": img, "pose_input": pose_feat}, label)


# =========================================================
# TRAIN DATASET
# =========================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_df["new_image_path"].values,
        X_pose_train.astype(np.float32),
        y_train.astype(np.float32),
    )
)

train_dataset = (
    train_dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# =========================================================
# TEST DATASET
# =========================================================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_df["new_image_path"].values,
        X_pose_test.astype(np.float32),
        y_test.astype(np.float32),
    )
)

test_dataset = (
    test_dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# =========================================================
# ALEXNET BRANCH
# AlexNet architecture (Krizhevsky et al., 2012):
#   Conv1: 96 filters, 11x11, stride 4  -> LRN -> MaxPool
#   Conv2: 256 filters,  5x5, pad same  -> LRN -> MaxPool
#   Conv3: 384 filters,  3x3, pad same
#   Conv4: 384 filters,  3x3, pad same
#   Conv5: 256 filters,  3x3, pad same  -> MaxPool
#   FC6:   4096
#   FC7:   4096
#   (FC8/output replaced by fusion head)
# =========================================================

cnn_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")

# --- Block 1 ---
x = Conv2D(96, kernel_size=11, strides=4, padding="valid", activation="relu")(cnn_input)
x = MaxPooling2D(pool_size=3, strides=2)(x)
x = BatchNormalization()(x)  # LRN approximated by BN

# --- Block 2 ---
x = Conv2D(256, kernel_size=5, strides=1, padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=3, strides=2)(x)
x = BatchNormalization()(x)

# --- Block 3 ---
x = Conv2D(384, kernel_size=3, strides=1, padding="same", activation="relu")(x)

# --- Block 4 ---
x = Conv2D(384, kernel_size=3, strides=1, padding="same", activation="relu")(x)

# --- Block 5 ---
x = Conv2D(256, kernel_size=3, strides=1, padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=3, strides=2)(x)

# --- Flatten & FC layers ---
x = Flatten()(x)

x = Dense(4096, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(4096, activation="relu")(x)
x = Dropout(0.5)(x)

# --- Feature vector for fusion (replaces FC8) ---
cnn_features = x  # shape: (batch, 4096)

# =========================================================
# IMAGE PROJECTION  (4096 -> 256)
# =========================================================

cnn_proj = Dense(256)(cnn_features)

cnn_proj = BatchNormalization()(cnn_proj)

cnn_proj = ReLU()(cnn_proj)

# =========================================================
# POSE FEATURE BRANCH
# =========================================================

pose_input = Input(shape=(X_pose_train.shape[1],), name="pose_input")

pose_branch = Dense(512)(pose_input)

pose_branch = BatchNormalization()(pose_branch)

pose_branch = ReLU()(pose_branch)

# ---------------------------------------------------------

pose_branch = Dense(512)(pose_branch)

pose_branch = BatchNormalization()(pose_branch)

pose_branch = ReLU()(pose_branch)

# =========================================================
# RESIDUAL CONNECTION
# =========================================================

pose_residual = pose_branch

# ---------------------------------------------------------

pose_branch = Dense(256)(pose_branch)

pose_branch = BatchNormalization()(pose_branch)

pose_branch = ReLU()(pose_branch)

# =========================================================
# PROJECT RESIDUAL
# =========================================================

pose_residual = Dense(256)(pose_residual)

# =========================================================
# ADD RESIDUAL
# =========================================================

pose_branch = tf.keras.layers.Add()([pose_branch, pose_residual])

pose_branch = ReLU()(pose_branch)

# =========================================================
# GATED FUSION
# =========================================================

cnn_gate = Dense(256, activation="sigmoid")(cnn_proj)

pose_gate = Dense(256, activation="sigmoid")(pose_branch)

# ---------------------------------------------------------

gated_cnn = Multiply()([cnn_proj, cnn_gate])

gated_pose = Multiply()([pose_branch, pose_gate])

# =========================================================
# FEATURE FUSION
# =========================================================

combined = Concatenate()([gated_cnn, gated_pose])

# =========================================================
# CLASSIFIER
# =========================================================

x = Dense(512)(combined)

x = BatchNormalization()(x)

x = ReLU()(x)

x = Dropout(0.3)(x)

# ---------------------------------------------------------

x = Dense(256)(x)

x = BatchNormalization()(x)

x = ReLU()(x)

x = Dropout(0.2)(x)

# =========================================================
# OUTPUT
# =========================================================

output = Dense(num_classes, activation="softmax")(x)

# =========================================================
# FINAL MODEL
# =========================================================

model = Model(inputs=[cnn_input, pose_input], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# =========================================================
# SAVE BEST MODEL  (.h5 format)
# =========================================================

checkpoint = ModelCheckpoint(
    filepath=os.path.join(BASE_DIR, "best_fusion_model_alexnet.h5"),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_format="h5",
    verbose=1,
)

# =========================================================
# CUSTOM LOGGER CALLBACK
# =========================================================


class EpochLogger(tf.keras.callbacks.Callback):

    def __init__(self, log_path):

        super().__init__()

        self.log_path = log_path

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        train_acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)

        train_loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)

        msg = (
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}\n"
        )

        print(msg)

        with open(self.log_path, "a", encoding="utf-8") as f:

            f.write(msg)


# =========================================================
# INITIALIZE LOGGER
# =========================================================

epoch_logger = EpochLogger(RESULT_LOG_PATH)

# =========================================================
# TRAIN
# =========================================================

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, epoch_logger],
)

# =========================================================
# PREDICT
# =========================================================

pred_probs = model.predict(test_dataset)

preds = np.argmax(pred_probs, axis=1)

# =========================================================
# OUTPUT + LOGS
# =========================================================

with open(RESULT_LOG_PATH, "a", encoding="utf-8") as f:

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("FINAL RESULTS\n")
    f.write("=" * 80 + "\n\n")

    # =====================================================
    # ACCURACY
    # =====================================================

    acc = accuracy_score(y_test_labels, preds)

    acc_msg = "\n" + "=" * 60 + "\n" f"TEST ACCURACY: {acc:.4f}\n" + "=" * 60 + "\n"

    print(acc_msg)

    f.write(acc_msg + "\n")

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    labels_present = np.unique(np.concatenate([y_test_labels, preds]))

    target_names_present = le.inverse_transform(labels_present)

    report = classification_report(
        y_test_labels,
        preds,
        labels=labels_present,
        target_names=target_names_present,
        zero_division=0,
    )

    print(report)

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 80 + "\n\n")

    f.write(report + "\n\n")

    # =====================================================
    # BEST EPOCH DETAILS
    # =====================================================

    best_epoch = np.argmax(history.history["val_accuracy"]) + 1

    best_acc = np.max(history.history["val_accuracy"])

    best_train_acc = history.history["accuracy"][best_epoch - 1]

    best_train_loss = history.history["loss"][best_epoch - 1]

    best_val_loss = history.history["val_loss"][best_epoch - 1]

    best_msg = (
        "\n" + "=" * 80 + "\n"
        "BEST MODEL DETAILS\n" + "=" * 80 + "\n\n"
        f"Best Epoch: {best_epoch}\n"
        f"Best Validation Accuracy: {best_acc:.4f}\n"
        f"Training Accuracy at Best Epoch: "
        f"{best_train_acc:.4f}\n"
        f"Training Loss at Best Epoch: "
        f"{best_train_loss:.4f}\n"
        f"Validation Loss at Best Epoch: "
        f"{best_val_loss:.4f}\n\n"
        f"Saved Best Model Path:\n"
        f"{checkpoint.filepath}\n"
    )

    print(best_msg)

    f.write(best_msg)

    # =====================================================
    # FULL EPOCH HISTORY
    # =====================================================

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("FULL TRAINING HISTORY\n")
    f.write("=" * 80 + "\n\n")

    for epoch in range(EPOCHS):

        train_acc = history.history["accuracy"][epoch]
        val_acc = history.history["val_accuracy"][epoch]

        train_loss = history.history["loss"][epoch]
        val_loss = history.history["val_loss"][epoch]

        epoch_msg = (
            f"Epoch {epoch+1:02d} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}\n"
        )

        f.write(epoch_msg)

print("\nLogs saved to:")
print(RESULT_LOG_PATH)
