# Prepare dataset for training

import os

import cv2

from utils import get_patches

TILE_SIZE = 256
print("Image patch size:", TILE_SIZE, "x", TILE_SIZE)

data_root = ""
gt_root = os.path.join(data_root, "GT")
img_list = os.listdir(data_root)
img_list.sort()

data_train_dir = os.path.join(data_root, "train")
if not os.path.exists(data_train_dir):
    os.makedirs(data_train_dir)

total_img_patches, total_msk_patches = [], []

possible_extensions = [".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"]
TILE_SIZE = 256

img_list = [f for f in os.listdir(data_root) if os.path.isfile(os.path.join(data_root, f)) and os.path.splitext(f)[1].lower() in possible_extensions]

img_list.sort()


for idx in range(len(img_list)):
    if os.path.isdir(os.path.join(data_root, img_list[idx])):
        continue

    print("Now processing image:", os.path.join(data_root, img_list[idx]))
    (fname, fext) = os.path.splitext(img_list[idx])
    img = cv2.imread(os.path.join(data_root, img_list[idx]))
    msk = None
    for ext in possible_extensions:
        msk_path = os.path.join(gt_root,fname + ext)
        if os.path.exists(msk_path):
            msk = cv2.imread(msk_path)
            if msk is not None:
                break

    img_patch_locations, img_patches = get_patches(img, TILE_SIZE, TILE_SIZE)
    msk_patch_locations, msk_patches = get_patches(msk, TILE_SIZE, TILE_SIZE)

    print("Patches extracted:", len(img_patches))
    for idy in range(len(img_patches)):
        total_img_patches.append(img_patches[idy])
        total_msk_patches.append(msk_patches[idy])

print("Total number of train patches:", len(total_img_patches))
for idz in range(len(total_img_patches)):
    cv2.imwrite(os.path.join(data_train_dir, str(idz) + "_img.png"), total_img_patches[idz])
    cv2.imwrite(os.path.join(data_train_dir, str(idz) + "_mask.png"), total_msk_patches[idz])

print("Done")
