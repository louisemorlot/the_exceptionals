import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
# from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

from skimage import io
import sys
sys.path.append("/localscratch/project/the_exceptionals/data/")
from transformation import sample_crops
sys.path.append("/localscratch/project/the_exceptionals/model/")
from helper import compute_receptive_field

from PIL import Image

def show_one_image(image_path):
    image = imageio.imread(image_path)
    plt.imshow(image)

class NumpyToTensor:
    def __call__(self, image):
        img_tensor = torch.from_numpy(image).float()
        return img_tensor
        
class ImageNormalize:
    def __call__(self, image):
        image = image.astype(np.float32)
        image = np.array(image)
        image = image / ((2**16-1)*1.0)
        #image = np.expand_dims(image, axis = 0)
        return image        

class ImageMinMaxNormalize:
    def __call__(self, image):
        image = image.astype(np.float32)
        image = np.array(image)
        image = ( image-np.min(image) )/ (np.max(image) - np.min(image) + 1e-6)
        image = np.expand_dims(image, axis = 0)
        return image 
        
class MaskNormalize:
    def __call__(self, mask):
        #mask = mask.astype(np.float32)
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis = 0)
        mask_tensor = torch.from_numpy(mask)
        return mask_tensor
    
class CellDataset(Dataset):
    """A Pytorch dataset to load the images and masks"""

    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(self.img_dir)
        self.masks = os.listdir(self.mask_dir)
        
        transform_img_list = []
        # transform_img_list += [transforms.Grayscale()]
        # transform_img_list += [transforms.ToTensor()] # already scales the image to [0,1]
        transform_img_list += [ImageNormalize()]
        transform_img_list += [ImageMinMaxNormalize()]
        transform_img_list += [NumpyToTensor()]

        transform_mask_list = []
        transform_mask_list += [MaskNormalize()]
        #transform_mask_list += [NumpyToTensor()]
        self.cropsize = 256
        self.weights = 1000000
        self.crop_flag = True
        self.img_transform = transforms.Compose(transform_img_list)
        self.mask_transform = transforms.Compose(transform_mask_list)

        #print (f"number of images: {len(self.images)}")

    # get the total number of samples
    def __len__(self):
        return len(self.images)*10

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        # image = Image.open(os.path.join(self.img_dir, self.images[idx%(len(self.images))]))
        # mask = Image.open(os.path.join(self.mask_dir, self.masks[idx%(len(self.masks))]))
        image = io.imread(os.path.join(self.img_dir, self.images[idx%(len(self.images))]))
        mask = io.imread(os.path.join(self.mask_dir, self.images[idx%(len(self.images))]))
        #print (f"image_shape: {image.shape}")
        #print (f"mask_shape: {mask.shape}")

        # Calculate crop coordinates from mask
        if self.crop_flag:
            cropCoords = sample_crops(mask, label_weight=self.weights, patch_size = [self.cropsize, self.cropsize])

            # Apply crop to image and mask
            image = image[cropCoords[0]:cropCoords[1],cropCoords[2]:cropCoords[3]]
            mask = mask[cropCoords[0]:cropCoords[1],cropCoords[2]:cropCoords[3]]
            mask[mask>0]=1
            mask = np.int8(mask)
        
        # Note: using seeds to ensure the same random transform is applied to
        # the image and mask
        seed = torch.seed()
        torch.manual_seed(seed)
        image = self.img_transform(image)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)
        return image, mask

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

class TestDataset(Dataset):
    """A Pytorch dataset to load the images and masks"""

    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(self.img_dir)
        self.masks = os.listdir(self.mask_dir)
        image = io.imread(os.path.join(self.img_dir, self.images[0]))
        mask = io.imread(os.path.join(self.mask_dir, self.images[0]))

        rfs = compute_receptive_field(depth = 3, kernel_size = 3, downsample_factor = 2)
        slide = int(rfs // 2)
        self.cropsize = 256
        
        image = image / ((2**16-1)*1.0)
        image = ( image-np.min(image) )/ (np.max(image) - np.min(image) + 1e-6)
        img_tensor = torch.from_numpy(image)
        img_crops = img_tensor.unfold(0, self.cropsize, slide).unfold(1, self.cropsize, slide)
        img_crops.unsqueeze(dim = 0)

        mask_tensor = torch.from_numpy(mask) # dim H, W
        mask_crops = mask_tensor.unfold(0, self.cropsize, slide).unfold(1, self.cropsize, slide)
        mask_crops.unsqueeze(dim = 0) # dim C, H, W
    

    # get the total number of samples
    def __len__(self):
        return len(self.images)*10

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        # image = Image.open(os.path.join(self.img_dir, self.images[idx%(len(self.images))]))
        # mask = Image.open(os.path.join(self.mask_dir, self.masks[idx%(len(self.masks))]))
        
        #print (f"image_shape: {image.shape}")
        #print (f"mask_shape: {mask.shape}")

        # Calculate receptive field size
        

        # Pad image with 0
        #img_pad = image.crop_pad((img_pad.shape[0], img_pad.shape[1]))
        #img_pad = torch.from_numpy(img_pad)

        # Crop full image into cropsized patches
        slide = int(rfs // 2)
        #slide = slide.to(torch.int)
        #img_crops = image.unfold(0, self.cropsize, slide).unfold(1, self.cropsize, slide)
        #img_crops = torch.flatten(img_crops, start_dim = 0, end_dim = 1)

       
        # Note: using seeds to ensure the same random transform is applied to
        # the image and mask
        seed = torch.seed()
        torch.manual_seed(seed)
        image = self.img_transform(image)
        img_crops = image.unfold(0, self.cropsize, slide).unfold(1, self.cropsize, slide)
        img_crops = torch.flatten(img_crops, start_dim = 0, end_dim = 1)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)
        return img_crops, mask
