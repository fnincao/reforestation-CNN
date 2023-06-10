from torch.utils.data import Dataset
import numpy as np
import rasterio
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


def normalize_image(image, sensor:str):
    image = image.astype(np.float32)
    if sensor == 'ndvi':
        image = ((image + 1) / 2)
    if sensor == 's1':
        image = np.clip(image, -50, 1)
        image = ((image + 50) / 51)
    if sensor == 'palsar':
        image = np.clip(image, 0, 10000)
        image = image / 10000
    return image


def gen_images(sensor: str, image_dir: str, mask_dir=None, transform=None,):
    random.seed(42)
    images = []
    masks = []
    img_files = sorted(glob.glob(image_dir + '/*' + sensor + '.tif'))

    if sensor == 'ndvi':
        mask_files = sorted(glob.glob(mask_dir + '/*tif'))
        for img, mask in zip(img_files, mask_files):
            with rasterio.open(img) as ds:
                image = np.transpose(ds.read(), (1, 2, 0))
                norm_img = normalize_image(image, sensor)

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

            transfomed = format_transform(image=norm_img, mask=mask)
            img_tensor = transfomed['image']
            mask_tensor = transfomed['mask']
            images.append(img_tensor)
            masks.append(mask_tensor)

            train_transform = A.Compose([
                A.Resize(height=image.shape[0], width=image.shape[1]),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1
                ),
                ToTensorV2(),
                ],)

            if transform:
                augmentations = train_transform(image=norm_img, mask=mask)
                trans_img = augmentations['image']
                trans_mask = augmentations['mask']
                images.append(trans_img)
                masks.append(trans_mask)

        return [images, masks]

    else:
        for img in img_files:
            with rasterio.open(img) as ds:
                image = np.transpose(ds.read(), (1, 2, 0))
                norm_img = normalize_image(image, sensor)

            format_transform = A.Compose([
                A.Resize(height=image.shape[0], width=image.shape[1]),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=1
                ),
                ToTensorV2(),
            ],)

            transfomed = format_transform(image=norm_img)
            img_tensor = transfomed['image']
            images.append(img_tensor)

            train_transform = A.Compose([
                A.Resize(height=image.shape[0], width=image.shape[1]),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1
                ),
                ToTensorV2(),
                ],)

            if transform:
                augmentations = train_transform(image=norm_img)
                trans_img = augmentations['image']
                images.append(trans_img)
        return images


class RSDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = gen_images('ndvi', image_dir, mask_dir, transform)
        self.s1 = gen_images('s1', image_dir, None, transform)
        self.palsar = gen_images('palsar', image_dir, None, transform)

    # Define len function
    def __len__(self):
        return len(self.images[1])

    def __getitem__(self, index):

        image = self.images[0][index]
        mask = self.images[1][index]
        s1 = self.s1[index]
        palsar = self.palsar[index]

        return image, s1, palsar, mask

