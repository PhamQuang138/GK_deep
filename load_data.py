import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(
        image,
        (128,128)
    )

    image = image.astype(np.float32) / 255.0

    # SAR log transform
    image = np.log1p(image)

    # normalize
    image = (
        image - image.mean()
    ) / (
        image.std() + 1e-8
    )

    # (H,W) -> (1,H,W)
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# =========================================================
# LOAD MASK
# =========================================================

def load_mask(path):

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.resize(
        mask,
        (128,128)
    )

    mask = mask.astype(np.float32) / 255.0

    # binary mask
    mask = (
        mask > 0.5
    ).astype(np.float32)

    # (H,W) -> (1,H,W)
    mask = np.expand_dims(
        mask,
        axis=0
    )

    return mask


# =========================================================
# BỘ GOM BATCH DỮ LIỆU (DATA GENERATOR)
# =========================================================

def data_generator(image_dir, mask_dir, batch_size=4):
    """
    Hàm đọc dữ liệu theo từng Batch để nạp vào file train.py
    """
    files = sorted(os.listdir(image_dir))
    num_samples = len(files)

    for i in range(0, num_samples, batch_size):
        batch_files = files[i:i + batch_size]

        batch_images = []
        batch_masks = []

        for filename in batch_files:
            img_path = os.path.join(image_dir, filename)
            msk_path = os.path.join(mask_dir, filename)

            # Đảm bảo mask tồn tại cùng tên với ảnh
            if os.path.exists(msk_path):
                batch_images.append(load_image(img_path))
                batch_masks.append(load_mask(msk_path))

        # Gộp list thành mảng 4D: (Batch_size, 1, 128, 128)
        yield np.array(batch_images), np.array(batch_masks)
# =========================================================
# DATASET PATH
# =========================================================

train_image_dir = "Deep_SAR/dataset/images/images/train"

train_mask_dir = "Deep_SAR/dataset/masks/masks/train"

val_image_dir = "Deep_SAR/dataset/images/images/val"

val_mask_dir = "Deep_SAR/dataset/masks/masks/val"


# =========================================================
# LOAD FILE LIST
# =========================================================

train_files = sorted(
    os.listdir(train_image_dir)
)

val_files = sorted(
    os.listdir(val_image_dir)
)


# =========================================================
# LOAD TRAIN SAMPLE
# =========================================================

filename = train_files[3500]

image_path = os.path.join(
    train_image_dir,
    filename
)

mask_path = os.path.join(
    train_mask_dir,
    filename
)

image = load_image(image_path)

mask = load_mask(mask_path)


# =========================================================
# CHECK TRAIN DATA
# =========================================================

print("========== TRAIN ==========")

print("Filename:", filename)

print("Image Shape:", image.shape)

print("Mask Shape :", mask.shape)

print("Image Min :", image.min())

print("Image Max :", image.max())

print("Mask Unique:", np.unique(mask))


# =========================================================
# SHOW TRAIN
# =========================================================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.imshow(
    image[0],
    cmap='gray'
)

plt.title("Train SAR")

plt.subplot(1,2,2)

plt.imshow(
    mask[0],
    cmap='gray'
)

plt.title("Train Mask")

plt.show()


# =========================================================
# LOAD VAL SAMPLE
# =========================================================

filename = val_files[300]

image_path = os.path.join(
    val_image_dir,
    filename
)

mask_path = os.path.join(
    val_mask_dir,
    filename
)

image = load_image(image_path)

mask = load_mask(mask_path)


# =========================================================
# CHECK VAL DATA
# =========================================================

print("\n========== VALIDATION ==========")

print("Filename:", filename)

print("Image Shape:", image.shape)

print("Mask Shape :", mask.shape)

print("Image Min :", image.min())

print("Image Max :", image.max())

print("Mask Unique:", np.unique(mask))


# =========================================================
# SHOW VAL
# =========================================================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.imshow(
    image[0],
    cmap='gray'
)

plt.title("Validation SAR")

plt.subplot(1,2,2)

plt.imshow(mask[0], cmap='gray')
plt.title("Validation Mask")
plt.show()