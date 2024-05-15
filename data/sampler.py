import os
import sys
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
sys.path.append("/localscratch/devel/the_exceptionals/data/")
from local import CellDataset
from torchvision import transforms


def index_from_pdf(pdf_im):
    prob = np.copy(pdf_im)
    # Normalize values to create a pdf with sum = 1
    prob = prob.ravel() / np.sum(prob)
    # Convert into a 1D pdf
    choices = np.prod(pdf_im.shape)
    index = np.random.choice(choices, size=1, p=prob)
    # Recover 2D shape
    coordinates = np.unravel_index(index, shape=pdf_im.shape)
    # Extract index
    indexh = coordinates[0][0]
    indexw = coordinates[1][0]
    return indexh, indexw


def sampling_pdf(y, pdf, height, width):
    h, w = y.shape[0], y.shape[1]
    #h, w = y.shape[0], y.shape[1]
    if pdf == 1:
        indexw = np.random.randint(np.floor(width // 2), \
                                   w - np.floor(width // 2))
        indexh = np.random.randint(np.floor(height // 2), \
                                   h - np.floor(height // 2))
    else:
        # Assign pdf values to foreground
        pdf_im = np.ones(y.shape, dtype=np.float32)
        pdf_im[y > 0] = pdf
        # crop to fix patch size
        pdf_im = pdf_im[np.int16(np.floor(height // 2)):-np.int16(np.floor(height // 2)), \
                 np.int16(np.floor(width // 2)):-np.int16(np.floor(width // 2))]
        indexh, indexw = index_from_pdf(pdf_im)
        indexw = indexw + np.int16(np.floor(width // 2))
        indexh = indexh + np.int16(np.floor(height // 2))

    return indexh, indexw

def show_random_sampler_image(sampleData):
    idx = np.random.randint(0, len(sampleData))  # take a random sample
    img = sampleData.loaded_input_patches[idx]
    mask = sampleData.loaded_masks_patches[idx]
    #img, mask = sampleData[idx]  # get the image and the nuclei masks
    f, axarr = plt.subplots(1, 2)  # make two plots on one figure
    axarr[0].imshow(img[0])  # show the image
    axarr[0].set_title("Image")
    axarr[1].imshow(mask[0], interpolation=None)  # show the masks
    axarr[1].set_title("Mask")
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    plt.show()

class Sampler():

    def __init__(self, image, mask):
        self.image = image
        self.mask = mask

    # get the total number of samples
    def __len__(self):
        return len(self.image)
        
    #def estimate_source_distribution(img_dir, mask_dir, transform = transforms_v2.RandomCrop(256)):
        #draw random distribution using CellDataset(img_dir = img_dir, mask_dir = mask_dir, transform = transforms_v2.RandomCrop(256)                           )
        #self.source_distribution

    def sample_crops(self, croping_weight=3000, patch_size = [256, 256]):
        # Sample training images to match the desired distribution from to min and max of the source distribution

        # Random draw, check target distribution 
        indexh, indexw = sampling_pdf(self.mask, croping_weight, patch_size[0], patch_size[1])
        lr = indexh - np.floor(patch_size[0] // 2)
        lr = lr.astype(np.int16)
        ur = indexh + np.round(patch_size[0] // 2)
        ur = ur.astype(np.int16)

        lc = indexw - np.floor(patch_size[1] // 2)
        lc = lc.astype(np.int16)
        uc = indexw + np.round(patch_size[1] // 2)
        uc = uc.astype(np.int16)

        mask_patch = self.mask[lr:ur, lc:uc]
        image_patch = self.image[lr:ur, lc:uc]
        #mask_patch = np.expand_dims(mask_patch, axis=0)
        #image_patch = np.expand_dims(image_patch, axis=0)
        #self.loaded_masks_patches[i] = transforms.ToTensor()(mask_patch)
        #self.loaded_input_patches[i] = transforms.ToTensor()(image_patch)
        return image_patch, mask_patch