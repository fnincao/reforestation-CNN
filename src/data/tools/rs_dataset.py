from torch.utils.data import Dataset
import numpy as np
import rasterio
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    if sensor == 'planet':
        image = np.clip(image, 0, 10000)
        image = image / 10000
    return image


def gen_images(sensor: str, image_dir: str, gen:bool):
    if gen:
        images = []
        img_files = sorted(glob.glob(image_dir + '/*' + sensor + '.tif'))
        for img in img_files:
            with rasterio.open(img) as ds:
                image = np.transpose(ds.read(), (1, 2, 0))
                norm_img = normalize_image(image, sensor)
                format_transform = A.Compose([
                A.Resize(height=image.shape[0], width=image.shape[1]),
                ToTensorV2()],)
                transfomed = format_transform(image=norm_img)
                img_tensor = transfomed['image']
                images.append(img_tensor)

        return images

class RSDataset(Dataset):
    def __init__(self, image_dir,
                 model:str,
                 ndvi=False,
                 s1=False,
                 palsar=False,
                 planet=False):
        self.image_dir = image_dir
        self.model = model
        self.ndvi = gen_images('ndvi', image_dir, ndvi)
        self.s1 = gen_images('s1', image_dir, s1)
        self.palsar = gen_images('palsar', image_dir, palsar)
        self.planet = gen_images('planet', image_dir, planet)

    # Define len function
    def __len__(self):
        if self.model == 'rgbn':
            return len(self.planet)
        
        if self.model == 'ndvi':
            return len(self.ndvi)
        
        if self.model == 'fusion':
            return len(self.ndvi)
        
    def __getitem__(self, index):
        
        if self.model == 'rgbn':
            image = self.planet[index]
            return image
        
        if self.model == 'ndvi':
            image = self.ndvi[index]
            return image
      
        if self.model == 'fusion':
            image = self.ndvi[index]
            s1 = self.s1[index]
            palsar = self.palsar[index]
            return image, s1, palsar

