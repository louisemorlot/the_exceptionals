# Library import
import tifffile
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch import no_grad, cuda
from transformation import augment_batch, normalize, denormalize
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision.transforms.v2 as transforms_v2

import sys
import os

sys.path.append("/localscratch/devel/the_exceptionals/model/")
from unet import UNet
from helper import train

sys.path.append("/localscratch/devel/the_exceptionals/util/")
from visualize import show_random_dataset_image_with_prediction

sys.path.append("/localscratch/devel/the_exceptionals/data/")
#from local import (
#    CellDataset,
#    show_random_dataset_image,
#    show_one_image
#)
import local

def train(img_dir, mask_dir, num_epochs=100, batch_size=5, shuffle=True, num_workers=8,
          depth=4, in_channels=1, out_channels=1, num_fmaps=64, transform=None):
    
    # Set device to gpu or cpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Import dataset
    #img_dir = "/localscratch/exceptionals/train_images2D/images"
    #mask_dir = "/localscratch/exceptionals/train_images2D/masks"
    
    trainData = local.CellDataset(img_dir = img_dir,
                            mask_dir = mask_dir
                           )
    
    #sampled_data = 

    # Start training
    train_loader= DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Create network
    unet = UNet(depth=depth, in_channels=in_channels, out_channels=out_channels, num_fmaps=num_fmaps, final_activation="Softmax").to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(unet.parameters())
    
    # Start training
    for epoch in range(num_epochs):
        print (f"epoch: {epoch}")
        run_training(unet, train_loader, optimizer, loss, epoch, device=device)
        
    return 


def run_training(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    device=None,
    early_stop=False,
):

    tb_logger = SummaryWriter("/localscratch/runs/Unet")
    
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        if prediction.shape != y.shape:
            y = crop(y, prediction)
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

                 
        # log to tensorboard
        step = epoch * len(loader) + batch_id
        
        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            
        # check if we log images in this iteration
        if step % log_image_interval == 0:
            tb_logger.add_images(
                tag="input", img_tensor=x.to("cpu"), global_step=step
            )
            tb_logger.add_images(
                tag="target", img_tensor=y.to("cpu"), global_step=step
            )
            tb_logger.add_images(
                tag="prediction",
                img_tensor=prediction.to("cpu").detach(),
                global_step=step,
            )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break
