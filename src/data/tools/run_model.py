from src.model_fusion.model import UNET as unet_fusion
from src.model_ndvi.model import UNET as unet_ndvi
from src.model_planet.model import UNET as unet_planet
from src.data.tools.rs_dataset import RSDataset
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
IMG_DIR = '../data/croped_data'
SAVE_DIR = '../data/predictions/'
CHECKPOINT_DIR = '../checkpoints/'

def normalize_image(image):
    # Convert the image to floating-point values
    image = image.astype(np.float32)
    # Normalize the image to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def segment_images(model_name:str, roi:int,
                  img_dir=IMG_DIR):
    
    if model_name == 'fusion':
        ds = RSDataset(img_dir, model=model_name,
                       ndvi=True, s1=True, palsar=True)
        model = unet_fusion(in_channels=3, out_channels=1).to(DEVICE) 
        if roi == 1:
            weights = CHECKPOINT_DIR + 'fusion.pth.tar'
        if roi == 2:
            weights = CHECKPOINT_DIR + 'fusion_ne.pth.tar'
    if model_name == 'ndvi':
        ds = RSDataset(img_dir, model=model_name, ndvi=True)
        model = unet_ndvi(in_channels=3, out_channels=1).to(DEVICE) 
        if roi == 1:
            weights = CHECKPOINT_DIR + 'ndvi.pth.tar'
        if roi == 2:
            weights = CHECKPOINT_DIR + 'ndvi_ne.pth.tar'
    if model_name == 'rgbn':
        ds = RSDataset(img_dir, model=model_name, planet=True)
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
            for x, y, z in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                z = z.to(DEVICE)
                preds = torch.sigmoid(model(x, y, z))
                preds = (preds > 0.5).float()
                numpy_array = preds.squeeze(dim=1).to('cpu').numpy()
                predictions.append(numpy_array)
    else:
    
        with torch.no_grad():
            for x in loader:
                x = x.to(DEVICE)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                numpy_array = preds.squeeze(dim=1).to('cpu').numpy()
                predictions.append(numpy_array)

    all_predictions = np.concatenate(predictions, axis=0)
    
    return all_predictions

    
def rgb_predictions(preds_fusion:None, preds_ndvi:None, preds_rgbn:None,
                    roi: int, img_dir=IMG_DIR):
    
    images = []
    files_planet = sorted(glob.glob(img_dir + '/*planet.tif'))
    for tile_number in range(len(files_planet)):
        with rasterio.open(files_planet[tile_number]) as planet_ds:
            planet = normalize_image(np.transpose(planet_ds.read()[0:3], (1, 2, 0)))
            planet = adjust_gamma(planet, 0.8)
            images.append(planet)

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

        fig, axs = plt.subplots(1, 3, figsize=(16, 16))

        axs[0].imshow(image_rgbn)
        axs[0].set_title('Prediction UNET RGBN')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        axs[1].imshow(image_ndvi)
        axs[1].set_title('Prediction UNET NDVI')
        axs[1].set_xticks([])
        axs[1].set_yticks([])

        axs[2].imshow(image_fusion)
        axs[2].set_title('Prediction UNET Fusion')
        axs[2].set_xticks([])
        axs[2].set_yticks([])

        tile_number = files_planet[idx].split('_')[-2]
        tile_string = f"tile_{tile_number}_roi{str(roi)}_preds_rgb.png"
        save_path = os.path.join(SAVE_DIR, tile_string)
        plt.savefig(save_path, facecolor='white', dpi=300, bbox_inches='tight')
        plt.close()

def save_geotiff(pred:np.ndarray, model:str, roi:int, img_dir=IMG_DIR):

    files_planet = sorted(glob.glob(img_dir + '/*planet.tif'))
    
    for idx, file in enumerate(files_planet):
        # Open the reference raster
        with rasterio.open(file) as ref_raster:
            # Get the metadata from the reference raster
            meta = ref_raster.meta.copy()

            # Update the metadata with the necessary details for the output GeoTIFF
            meta.update(
                driver='GTiff',
                dtype='float32',
                count=1,  # Number of bands in the output GeoTIFF (1 for grayscale)
                nodata=None  # Set this to a specific value if applicable, otherwise None
            )
            tile_number = files_planet[idx].split('_')[-2]
            tile_string = f"tile_{tile_number}_roi{str(roi)}_pred.tif"
            output_path = SAVE_DIR + tile_string
            # Save the array as a GeoTIFF file
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(pred[idx], 1)  # Write the array to the first band of the output GeoTIFF
