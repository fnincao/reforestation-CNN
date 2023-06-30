from src.model_fusion.model import UNET as unet_fusion
from src.model_ndvi.model import UNET as unet_ndvi
from src.model_planet.model import UNET as unet_planet
from src.model_fusion.dataset import RSDataset as ds_fusion
from src.model_planet.dataset import PlanetDataset as ds_planet
from src.model_ndvi.dataset import PlanetDataset as ds_ndvi
from torch.utils.data import DataLoader
from skimage.exposure import rescale_intensity, adjust_gamma
import cv2
import matplotlib.pyplot as plt
import rasterio
import glob
import numpy as np
import torch
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = '../data/ai_data/val_images'
MASK_DIR = '../data/ai_data/val_masks'
SAVE_DIR = '../data/predictions/'
CHECKPOINT_DIR = '../checkpoints/'

def normalize_image(image):
    # Convert the image to floating-point values
    image = image.astype(np.float32)
    # Normalize the image to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def segment_images(model_name:str, roi:int,
                  img_dir=IMG_DIR, mask_dir=MASK_DIR):
    
    if model_name == 'fusion':
        ds = ds_fusion(image_dir=img_dir,
                       mask_dir=mask_dir,
                       transform=None)
        model = unet_fusion(in_channels=3, out_channels=1).to(DEVICE) 
        if roi == 1:
            weights = CHECKPOINT_DIR + 'fusion.pth.tar'
        if roi == 2:
            weights = CHECKPOINT_DIR + 'fusion_ne.pth.tar'
    if model_name == 'ndvi':
        ds = ds_ndvi(image_dir=img_dir,
                     mask_dir=mask_dir,
                     transform=None)
        channels = 3
        model = unet_ndvi(in_channels=3, out_channels=1).to(DEVICE) 
        if roi == 1:
            weights = CHECKPOINT_DIR + 'ndvi.pth.tar'
        if roi == 2:
            weights = CHECKPOINT_DIR + 'ndvi_ne.pth.tar'
    if model_name == 'rgbn':
        ds = ds_planet(image_dir=img_dir,
                       mask_dir=mask_dir,
                       transform=None)
        model = unet_planet(in_channels=4, out_channels=1).to(DEVICE) 
        if roi == 1:
            weights = CHECKPOINT_DIR + 'planet.pth.tar'
        if roi == 2:
            weights = CHECKPOINT_DIR + 'planet_ne.pth.tar'   
 
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])
 
    loader = DataLoader(ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    shuffle=False)
    
    predictions = []
    model.eval()
    
    if model_name == 'fusion':
        with torch.no_grad():
            for x, y, z, m in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                z = z.to(DEVICE)
                m = m.to(DEVICE).unsqueeze(1)
                preds = torch.sigmoid(model(x, y, z))
                preds = (preds > 0.5).float()
                numpy_array = preds.squeeze(dim=1).to('cpu').numpy()
                predictions.append(numpy_array)
    else:
    
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                numpy_array = preds.squeeze(dim=1).to('cpu').numpy()
                predictions.append(numpy_array)

    all_predictions = np.concatenate(predictions, axis=0)
    
    return all_predictions

    
def vis_predictions(preds_fusion:None, preds_ndvi:None, preds_rgbn:None,
                    roi: int, img_dir=IMG_DIR, mask_dir=MASK_DIR ):
    
    images = []
    files_planet = sorted(glob.glob(img_dir + '/*planet.tif'))
    for tile_number in range(len(files_planet)):
        with rasterio.open(files_planet[tile_number]) as planet_ds:
            planet = normalize_image(np.transpose(planet_ds.read()[0:3], (1, 2, 0)))
            planet = adjust_gamma(planet, 0.8)
            images.append(planet)
    
    gd_truth = []
    files_ref = sorted(glob.glob(mask_dir + '/*ref.tif'))
    for tile_number in range(len(files_ref)):
        with rasterio.open(files_ref[tile_number]) as ref_ds:
            ref = ref_ds.read(1).astype(np.uint8)
            gd_truth.append(ref)

    for idx in range(len(images)):
        image_planet = images[idx]
        seg_fusion = preds_fusion[idx].astype(np.uint8)
        seg_ndvi = preds_ndvi[idx].astype(np.uint8)
        seg_rgbn = preds_rgbn[idx].astype(np.uint8)

        edges_fusion = cv2.Canny(seg_fusion, threshold1=0, threshold2=1)
        red_mask_fusion = np.stack((edges_fusion,) * 3, axis=-1)
        image_fusion = np.where(red_mask_fusion > 0, (1, 0, 0), image_planet)
        
        edges_ndvi = cv2.Canny(seg_ndvi, threshold1=0, threshold2=1)
        red_mask_ndvi = np.stack((edges_ndvi,) * 3, axis=-1)
        image_ndvi = np.where(red_mask_ndvi > 0, (1, 0, 0), image_planet)

        edges_rgbn = cv2.Canny(seg_rgbn, threshold1=0, threshold2=1)
        red_mask_rgbn = np.stack((edges_rgbn,) * 3, axis=-1)
        image_rgbn = np.where(red_mask_rgbn > 0, (1, 0, 0), image_planet)
        
        reference = gd_truth[idx]
        edges_ref = cv2.Canny(reference, threshold1=0, threshold2=1)
        red_mask_ref = np.stack((edges_ref,) * 3, axis=-1)
        image_ref = np.where(red_mask_ref > 0, (1, 0, 0), image_planet)

        fig, axs = plt.subplots(1, 4, figsize=(16, 16))
        axs[0].imshow(image_ref)
        axs[0].set_title(f'Ground Truth')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].imshow(image_rgbn)
        axs[1].set_title('Prediction UNET RGBN')
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        axs[2].imshow(image_ndvi)
        axs[2].set_title('Prediction UNET NDVI')
        axs[2].set_xticks([])
        axs[2].set_yticks([])

        axs[3].imshow(image_fusion)
        axs[3].set_title('Prediction UNET Fusion')
        axs[3].set_xticks([])
        axs[3].set_yticks([])

        tile_number = files_planet[idx].split('_')[-2]
        tile_string = f"tile_{tile_number}_{str(roi)}_preds.png"
        save_path = os.path.join(SAVE_DIR, tile_string)
        plt.savefig(save_path, facecolor='white', dpi=300, bbox_inches='tight')
        plt.close()
