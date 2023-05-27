import os
from torch.utils.data import Dataset
import numpy as np
import rasterio


class PlanetDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    # Define len function
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir,
                                 self.images[index].replace("_planet.tif", "_ref.tif")) # noqa
        with rasterio.open(img_path) as ds:
            ds_nir = ds.read(1)
            ds_red = ds.read(2)
            ds_green = ds.read(3)
            image = np.stack([ds_nir, ds_red, ds_green], axis=-1)
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
