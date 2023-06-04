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


def stack_images(image_dir: str, mask_dir: str, transform=None):
    images = []
    masks = []
    img_files = sorted(glob.glob(image_dir + '/*planet.tif'))
    mask_files = sorted(glob.glob(mask_dir + '/*tif'))

    for img, mask in zip(img_files, mask_files):

        with rasterio.open(img) as ds:

            ds_nir = normalize_image(ds.read(1))
            ds_red = normalize_image(ds.read(2))
            ds_green = normalize_image(ds.read(3))
            image = (np.stack([ds_nir, ds_red, ds_green], axis=-1))

        with rasterio.open(mask) as ds:
            mask = ds.read(1).astype(float)

        format_transform = A.Compose([
            A.Resize(height=image.shape[0], width=image.shape[1]),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
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


class PlanetDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = stack_images(image_dir, mask_dir, transform)

    # Define len function
    def __len__(self):
        return len(self.images[1])

    def __getitem__(self, index):
        image = self.images[0][index]
        mask = self.images[1][index]

        return image, mask
