import os
import imageio
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.segmentation import relabel_sequential
from scipy.optimize import linear_sum_assignment

def show_random_dataset_image(dataset):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    f, axarr = plt.subplots(1, 2)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[0].set_title("Image")
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    axarr[1].set_title("Mask")
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()


def show_random_dataset_image_with_prediction(dataset, model, device="cpu"):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    x = img.to(device).unsqueeze(0)
    y = model(x)[0].detach().cpu().numpy()
    print("MSE loss:", np.mean((mask[0].numpy() - y[0]) ** 2))
    f, axarr = plt.subplots(1, 3)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[0].set_title("Image")
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    axarr[1].set_title("Mask")
    axarr[2].imshow(y[0], interpolation=None)  # show the prediction
    axarr[2].set_title("Prediction")
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()


def show_random_augmentation_comparison(dataset_a, dataset_b):
    assert len(dataset_a) == len(dataset_b)
    idx = np.random.randint(0, len(dataset_a))  # take a random sample
    img_a, mask_a = dataset_a[idx]  # get the image and the nuclei masks
    img_b, mask_b = dataset_b[idx]  # get the image and the nuclei masks
    f, axarr = plt.subplots(2, 2)  # make two plots on one figure
    axarr[0, 0].imshow(img_a[0])  # show the image
    axarr[0, 0].set_title("Image")
    axarr[0, 1].imshow(mask_a[0], interpolation=None)  # show the masks
    axarr[0, 1].set_title("Mask")
    axarr[1, 0].imshow(img_b[0])  # show the image
    axarr[1, 0].set_title("Augmented Image")
    axarr[1, 1].imshow(mask_b[0], interpolation=None)  # show the prediction
    axarr[1, 1].set_title("Augmented Mask")
    _ = [ax.axis("off") for ax in axarr.flatten()]  # remove the axes
    plt.show()


def apply_and_show_random_image(f, ds):

    # pick random raw image from dataset
    img_tensor = ds[np.random.randint(len(ds))][0]

    batch_tensor = torch.unsqueeze(
        img_tensor, 0
    )  # add batch dimension that some torch modules expect
    out_tensor = f(batch_tensor)  # apply torch module
    out_tensor = out_tensor.squeeze(0)  # remove batch dimension
    img_arr = img_tensor.numpy()[0]  # turn into numpy array, look at first channel
    out_arr = out_tensor.detach().numpy()[
        0
    ]  # turn into numpy array, look at first channel

    # intialilze figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 20))

    # Show input image, add info and colorbar
    img_min, img_max = (img_arr.min(), img_arr.max())  # get value range
    inim = axs[0].imshow(img_arr, vmin=img_min, vmax=img_max)
    axs[0].set_title("Input Image")
    axs[0].set_xlabel(f"min: {img_min:.2f}, max: {img_max:.2f}, shape: {img_arr.shape}")
    div = make_axes_locatable(axs[0])
    cb = fig.colorbar(inim, cax=div.append_axes("right", size="5%", pad=0.05))
    cb.outline.set_visible(False)

    # Show ouput image, add info and colorbar
    out_min, out_max = (out_arr.min(), out_arr.max())  # get value range
    outim = axs[1].imshow(out_arr, vmin=out_min, vmax=out_max)
    axs[1].set_title("First Channel of Output")
    axs[1].set_xlabel(f"min: {out_min:.2f}, max: {out_max:.2f}, shape: {out_arr.shape}")
    div = make_axes_locatable(axs[1])
    cb = fig.colorbar(outim, cax=div.append_axes("right", size="5%", pad=0.05))
    cb.outline.set_visible(False)

    # center images and remove ticks
    max_bounds = [
        max(ax.get_ybound()[1] for ax in axs),
        max(ax.get_xbound()[1] for ax in axs),
    ]
    for ax in axs:
        diffy = abs(ax.get_ybound()[1] - max_bounds[0])
        diffx = abs(ax.get_xbound()[1] - max_bounds[1])
        ax.set_ylim([ax.get_ybound()[0] - diffy / 2.0, max_bounds[0] - diffy / 2.0])
        ax.set_xlim([ax.get_xbound()[0] - diffx / 2.0, max_bounds[1] - diffx / 2.0])
        ax.set_xticks([])
        ax.set_yticks([])

        # for spine in ["bottom", "top", "left", "right"]: # get rid of box
        #     ax.spines[spine].set_visible(False)

