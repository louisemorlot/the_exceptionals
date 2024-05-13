import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def show_one_image(image_path):
    image = imageio.imread(image_path)
    plt.imshow(image)

class CellDataset(Dataset):
    """A Pytorch dataset to load the images and masks"""

    def __init__(self, img_dir, mask_dir, transform = None, img_transform = None):
        self.img_dir = ("/louise.morlot/course/project_dl4mia/devel/" + img_dir)
        self.mask_dir = ("/louise.morlot/course/project_dl4mia/devel/" + mask_dir)
        self.images = os.listdir(self.img_dir)
        self.masks = os.listdir(self.mask_dir)

        self.img_transform = img_transform

        inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([self.mean],[self.std]),
            ]
        )

        self.loaded_imgs = [None] * len(self.images)
        self.loaded_masks = [None] * len(self.mask)

        for img_ind in range(len(self.images)):
            img_path = os.path.join(
                self.img_dir, self.images[img_ind], ".tif"
            )
            image = Image.open(img_path)
            image.load()
            self.mean = image.mean()
            self.std = image.std()
            self.loaded_imgs[sample_ind] = inp_transforms(image)

        for mask_ind in range(len(self.masks)):
            mask_path = os.path.join(
                self.mask_dir, self.masks[mask_ind], ".tif"
            )
            mask = Image.open(mask_path)
            mask.load()
            self.loaded_masks[sample_ind] = transforms.ToTensor()(mask)

    # get the total number of samples
    def __len__(self):
        return len(self.images)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
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
