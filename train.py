import os
from time import time

import torch
import torch.utils.data as data

from data import ImageFolder
from framework import Frame
from loss import dice_bce_mae_loss
from networks.MRANet import MRANet


SHAPE = (256, 256)
DATA_NAME = ""
DEEP_NETWORK_NAME = ""  #MRANet
print("Now training dataset: {}, using network model: {}".format(DATA_NAME, DEEP_NETWORK_NAME))

train_root = ""
imagelist = list(filter(lambda x: x.find("img") != -1, os.listdir(train_root)))
trainlist = list(map(lambda x: x[:-8], imagelist))
log_name = DATA_NAME.lower() + "_" + DEEP_NETWORK_NAME.lower()

BATCHSIZE_PER_CARD = 32  #32

if DEEP_NETWORK_NAME == "MRANet":
    solver = MyFrame(MRANet, dice_bce_mae_loss)
elif DEEP_NETWORK_NAME == "MRANet":
    solver = MyFrame(MRANet, dice_bce_mae_loss)
else:
    print("Deep network not found, please have a check!")
    exit(0)

batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
total_epoch = 100

dataset = ImageFolder(trainlist, train_root)
from torch.utils.data import random_split


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4)

val_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batchsize,
    shuffle=False,  
    num_workers=4)


train_epoch_best_iou = -float('inf')  
mylog = open("logs/" + log_name + ".log", "w")
no_optim = 0
tic = time()


model_save_dir = os.path.join("weights", log_name)
os.makedirs(model_save_dir, exist_ok=True)  



train_epoch_best_f1 = -float('inf') 


for epoch in range(1, total_epochs + 1):
    print(f"Epoch {epoch}/{total_epochs}")
    print("-" * 10)

    train_epoch_loss = 0.0
    val_epoch_loss = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    start_time = time.time()

    model.train()
    for img, mask in train_data_loader:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss

    train_epoch_loss /= len(train_data_loader)

    model.eval()
    with torch.no_grad():
        for img, mask in val_data_loader:
            solver.set_input(img, mask)
            outputs = solver.get_output()
            mask = mask.to(outputs.device)
            val_loss = solver.loss(mask, outputs)
            val_epoch_loss += val_loss

            predicted = (outputs > 0.5).float()
            mask = mask.float()
            predicted = predicted.view(-1)
            mask = mask.view(-1)

            tp = ((predicted == 1) & (mask == 1)).sum().item()
            fp = ((predicted == 1) & (mask == 0)).sum().item()
            fn = ((predicted == 0) & (mask == 1)).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

    val_epoch_loss /= len(val_data_loader)
    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall = total_tp / (total_tp + total_fn + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Logging
    print(f"train_loss: {train_epoch_loss:.4f}", file=mylog)
    print(f"val_loss: {val_epoch_loss:.4f}, val_f1: {f1_score:.4f}", file=mylog)
    print(f"train_loss: {train_epoch_loss:.4f}")
    print(f"val_loss: {val_epoch_loss:.4f}, val_f1: {f1_score:.4f}")

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch time: {epoch_time:.2f} seconds\n", file=mylog)
    print(f"Epoch time: {epoch_time:.2f} seconds\n")
