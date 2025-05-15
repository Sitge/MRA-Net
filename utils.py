import numpy as np
import cv2

TILE_SIZE = 128
PADDING_SIZE = 21

LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2


def get_patches(img, patch_h=TILE_SIZE, patch_w=TILE_SIZE):
    y_stride, x_stride = patch_h - (2 * PADDING_SIZE), patch_w - (2 * PADDING_SIZE)

    if patch_h > img.shape[0] or patch_w > img.shape[1]:
        pad_h = max(0, patch_h - img.shape[0])
        pad_w = max(0, patch_w - img.shape[1])
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
        print(f"Image padded to shape: {img.shape}")
    
    if img.shape[0] % patch_h != 0:
        pad_h = patch_h - (img.shape[0] % patch_h)
        img = np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode='constant', constant_values=255)

    if img.shape[1] % patch_w != 0:
        pad_w = patch_w - (img.shape[1] % patch_w)
        img = np.pad(img, ((0, 0), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
    
    print(f"Image final padded shape: {img.shape}")

    locations, patches = [], []
    y = 0
    y_done = False
    
    while y <= img.shape[0] and not y_done:
        x = 0
        if y + patch_h > img.shape[0]:
            y = img.shape[0] - patch_h
            y_done = True
        x_done = False
        while x <= img.shape[1] and not x_done:
            if x + patch_w > img.shape[1]:
                x = img.shape[1] - patch_w
                x_done = True
            locations.append(((y, x, y + patch_h, x + patch_w),
                              (y + PADDING_SIZE, x + PADDING_SIZE, y + y_stride, x + x_stride),
                              TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (img.shape[0] - patch_h) else MIDDLE),
                              LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (img.shape[1] - patch_w) else MIDDLE)))
            patches.append(img[y:y + patch_h, x:x + patch_w, :])
            x += x_stride
        y += y_stride

    return locations, patches





def stitch_together(locations, patches, size, patch_h=TILE_SIZE, patch_w=TILE_SIZE):
    if len(patches[0].shape) == 3:
        output = np.zeros((size[0], size[1], patches[0].shape[2]), dtype=np.float32)
    else:
        output = np.zeros(size, dtype=np.float32)

    for i, (location, patch) in enumerate(zip(locations, patches)):
        outer_bounding_box, inner_bounding_box, y_type, x_type = location

        y_paste, x_paste = outer_bounding_box[0], outer_bounding_box[1]

        patch_h_paste = min(patch_h, size[0] - y_paste)
        patch_w_paste = min(patch_w, size[1] - x_paste)

        patch = patch[:patch_h_paste, :patch_w_paste]

        output[y_paste:y_paste + patch_h_paste, x_paste:x_paste + patch_w_paste] = patch
        
        print(f"Patch {i}: Paste to ({y_paste}, {x_paste}) with size ({patch_h_paste}, {patch_w_paste})")

    return output
