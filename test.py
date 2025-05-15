import os
from time import time

import cv2
import numpy as np
import torch
from torch.autograd import Variable as V

from utils import get_patches, stitch_together
from networks.MRANet import MRANet

BATCHSIZE_PER_CARD = 256


class Frame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_img(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        img = np.array(path)
        img = img.transpose(2, 0, 1)[None]
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        mask = self.net.forward(img).squeeze().cpu().data.numpy()
        return mask

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


TILE_SIZE = 256
DATA_NAME = ""  # HanBamboo
DEEP_NETWORK_NAME = ""  # MRANet
epoch_num = 200

img_indir = ""
print("Image input directory:", img_indir)
img_outdir = os.path.join(img_indir, "Binarized" + "_" + DEEP_NETWORK_NAME + "_epoch" + str(epoch_num))
if not os.path.exists(img_outdir):
    os.makedirs(img_outdir)
print("Image output directory:", img_outdir)

img_list = os.listdir(img_indir)
img_list.sort()

if DEEP_NETWORK_NAME == "MRANet":
    solver = TTAFrame(MRANet)
elif DEEP_NETWORK_NAME == "MRANet":
    solver = TTAFrame(MRANet)
else:
    print("Deep network not found, please have a check!")
    exit(0)


solver.load("weights/" +DATA_NAME.lower() + "_" + DEEP_NETWORK_NAME.lower() +"/"+ DATA_NAME.lower() + "_" + DEEP_NETWORK_NAME.lower() +"_epoch"+str(epoch_num) +".th")
print("Now loading the model weights:", "weights/" + DATA_NAME.lower() + "_" + DEEP_NETWORK_NAME.lower() + ".pth")


start_time = time()
for idx in range(len(img_list)):
    if os.path.isdir(os.path.join(img_indir, img_list[idx])):
        continue

    print("Now processing image:", img_list[idx])
    fname, fext = os.path.splitext(img_list[idx])
    img_input = os.path.join(img_indir, img_list[idx])
    img_output = os.path.join(img_outdir, fname + "-" + DEEP_NETWORK_NAME + "_epoch" + str(epoch_num) + ".png")


    img = cv2.imread(img_input)
    locations, patches = get_patches(img, TILE_SIZE, TILE_SIZE)
    masks = []
    for idy in range(len(patches)):
        msk = solver.test_img(patches[idy])
        masks.append(msk)
    prediction = stitch_together(locations, masks, tuple(img.shape[0:2]), TILE_SIZE, TILE_SIZE)
    print("prediction",prediction)
    prediction[prediction >= 0.5] = 255
    prediction[prediction < 0.5] = 0
    cv2.imwrite(img_output, prediction.astype(np.uint8))

print("Total running time: %f sec." % (time() - start_time))
print("Finished!")
