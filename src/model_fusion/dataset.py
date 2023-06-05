from torch.utils.data import Dataset
import numpy as np
import rasterio
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

random.seed(42)
np.random.seed(42)


def normalize_image(image):
    # Convert the image to floating-point values
    image = image.astype(np.float32)
    # Normalize the image to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def gen_images(sensor:str, image_dir: str, mask_dir=None, transform=None,):
    random.seed(42)
    images = []
    masks = []
    img_files = sorted(glob.glob(image_dir + sensor + '.tif')
    if sensor == 'red':
        mask_files = sorted(glob.glob(mask_dir + '/*tif'))
        for img, mask in zip(img_files, mask_files):
            with rasterio.open(img) as ds:
                image = np.transpose(ds.read(), (1, 2, 0))
                norm_img = normalize_image(image)

            with rasterio.open(mask_path) as ds:
                mask = ds.read(1).astype(float)

            format_transform = A.Compose([
                A.Resize(height=image.shape[0], width=image.shape[1]),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0, 1.0, 1.0],
                    max_pixel_value=1

                ),
                ToTensorV2(),
            ],)

            transfomed = format_transform(image=image, mask=mask)
            img_tensor = transfomed['image']
            mask_tensor = transfomed['mask']
            images.append(img_tensor)
            masks.append(mask_tensor)

            if transform is not None:
                augmentations = transform(image=image, mask=mask)
                trans_img = augmentations['image']
                trans_mask = augmentations['mask']
                images.append(trans_img)
                masks.append(trans_mask)

        return [images, masks]

    else:
        for img in img_files:
            with rasterio.open(img) as ds:
                image = np.transpose(ds.read(), (1, 2, 0))
                norm_img = normalize_image(image)
                       
            format_transform = A.Compose([
                A.Resize(height=image.shape[0], width=image.shape[1]),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0, 1.0, 1.0],
                    max_pixel_value=1
                ),
                ToTensorV2(),
            ],)

            transfomed = format_transform(image=image)
            img_tensor = transfomed['image']
            images.append(img_tensor)

            if transform is not None:
                augmentations = transform(image=image)
                trans_img = augmentations['image']
                images.append(trans_img)

        return images

        
class PlanetDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 transform=None,
                 sensor='red'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = gen_images(sensor, image_dir, mask_dir, transform)

    # Define len function
    def __len__(self):
        return len(self.images[1])

    def __getitem__(self, index):

        image = self.images[0][index]
        mask = self.images[1][index]

        return image, mask

                       
class SentinelDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 transform=None,
                 sensor='s1'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = gen_images(sensor, image_dir, mask_dir, transform)

    # Define len function
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        return image
                       

class PalsarDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 transform=None,
                 sensor='palsar'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = gen_images(sensor, image_dir, mask_dir, transform)

    # Define len function
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        return image
