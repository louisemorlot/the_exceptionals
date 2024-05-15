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

class Global_normalize(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img):
        img_normd = img / ((2**16-1)*1.0)
        return img_normd 

class CellDataset(Dataset):
    """A Pytorch dataset to load the images and masks"""

    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(self.img_dir)
        self.masks = os.listdir(self.mask_dir)

        transform_list = []
        transform_list += [transforms.Grayscale()]
        transform_list += [transforms.ToTensor()]
        transform_list += [Global_normalize()]
        # transform_list += [transforms.Lambda(lambda  img: self.__normalize(img))]
        self.transform_a = transforms.Compose(transform_list)

        # self.loaded_imgs = [None] * len(self.images)
        # self.loaded_masks = [None] * len(self.masks)

        print (f"number of images: {len(self.images)}")
        # for img_ind in range(len(self.images)):
        #     img_path = os.path.join(
        #         self.img_dir, self.images[img_ind]
        #     )
        #     image = Image.open(img_path)
        #     image.load()
        #     #self.mean = image.mean()
        #     #self.std = image.std()
        #     self.loaded_imgs[img_ind] = self.transform_a(image)

        # for mask_ind in range(len(self.masks)):
        #     mask_path = os.path.join(
        #         self.mask_dir, self.masks[mask_ind]
        #     )
        #     mask = Image.open(mask_path)
        #     mask.load()
        #     self.loaded_masks[mask_ind] = transforms.ToTensor()(mask)

    # get the total number of samples
    def __len__(self):
        return len(self.images)*10

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # we'll be using Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = Image.open(os.path.join(self.img_dir, self.images[idx%(len(self.images))]))
        image.load()

        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx%(len(self.masks))]))
        mask.load()
        # Note: using seeds to ensure the same random transform is applied to
        # the image and mask
        seed = torch.seed()
        torch.manual_seed(seed)
        image = self.transform_a(image)
        torch.manual_seed(seed)
        mask = self.transform_a(mask)
        # if self.img_transform is not None:
        #     image = self.img_transform(image)
        return image, mask

    def __normalize(self, img):
        img_norm = img / ((2**16-1)*1.0)
        return img_norm
    
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
