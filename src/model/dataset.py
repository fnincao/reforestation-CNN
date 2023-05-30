import os
from torch.utils.data import Dataset
import numpy as np
import rasterio


def normalize_image(image):
    # Convert the image to floating-point values
    image = image.astype(np.float32)
    # Normalize the image to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


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
        planet_path = os.path.join(self.image_dir, self.images[index])
        
        s1_path = os.path.join(self.s1_dir,
                               self.images[index].replace("_planet.tif",
                                                          "_s1.tif"))
        s2_path = os.path.join(self.s2_dir,
                               self.images[index].replace("_planet.tif",
                                                          "_s2.tif"))
        mask_path = os.path.join(self.mask_dir,
                                 self.images[index].replace("_planet.tif",
                                                            "_ref.tif"))
        
        
        with rasterio.open(planet_path) as ds,
             rasterio.open(s1_path) as ds_s1,
             rasterio.open(s2_path) as ds_s2:
                    
            ds_nir = normalize_image(ds.read(1))
            ds_red = normalize_image(ds.read(2))
            ds_green = normalize_image(ds.read(3))
            
            ds_sar = normalize_image(ds_s1.read(1))
            
            ds_s2_2020 = normalize_image(ds_s2.read(1))
            ds_s2_2021 = normalize_image(ds_s2.read(2))
            ds_s2_2022 = normalize_image(ds_s2.read(3))
            
            image = (np.stack([ds_nir, ds_red, ds_green, ds_sar,
                              ds_s2_2020, ds_s2_2020, ds_s2_2022], axis=-1))
            
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1).astype(float)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
