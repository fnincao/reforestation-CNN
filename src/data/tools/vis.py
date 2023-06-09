'''Module to vizualize image chips downloaded from GEE'''

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity, adjust_gamma
import glob
import os
import cv2


plt.rcParams['figure.dpi'] = 300


def normalize_image(image):
    # Convert the image to floating-point values
    image = image.astype(np.float32)
    # Normalize the image to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image



def viz_reference(tile_number: int, save_fig:bool):
    '''
    Visualizes a reference imagery tile.

    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    viz_reference(tile_number=0, save_fig=True)
    
    Note:
    - Assumes that the reference files are in GeoTIFF format.
    - The visualization is saved in '../../../data/figures/'.
    '''
    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    tile_str = os.path.splitext(os.path.basename(files_ref[tile_number]))[0]
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_ref[tile_number]) as reference:
        reference_data = reference.read(1).astype(np.uint8)
        plt.imshow(reference_data, vmin=0, vmax=1, cmap='gray')
        plt.yticks([])
        plt.xticks([])
        
        if save_fig:
            save_path = os.path.join(save_dir, f"{tile_str}.png")
            plt.imsave(save_path, reference_data, cmap='gray')
            
        plt.title(tile_str)


def viz_planet(tile_number: int, draw_ref: bool, save_fig: bool):
    '''
    Visualizes a Planet RGB imagery tile.
    RGB median composite from year 2020.

    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - draw_ref (bool): Flag to draw reference borders on the image.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    viz_planet(tile_number=0, draw_ref=True, save_fig=True)
    
    Note:
    - Assumes that the planet and reference files are in GeoTIFF format.
    - The visualization is saved in '../../../data/figures/'.
    '''

    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_planet = sorted(glob.glob(os.path.join(imgs_dir, '*planet.tif')))
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    tile_str = os.path.splitext(os.path.basename(files_planet[tile_number]))[0]
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_planet[tile_number]) as planet, \
         rasterio.open(files_ref[tile_number]) as reference:
        image = normalize_image(np.transpose(planet.read(), (1, 2, 0)))
        gamma = adjust_gamma(image, 0.8)
        ref = reference.read(1).astype(np.uint8)

        if draw_ref:
            edges = cv2.Canny(ref, threshold1=0, threshold2=1)
            red_mask = np.stack((edges,) * 3, axis=-1)
            gamma = np.where(red_mask > 0, (1, 0, 0), gamma)

        plt.imshow(gamma)
        plt.yticks([])
        plt.xticks([])

        if save_fig:
            save_path = os.path.join(save_dir, f"{tile_str}.png")
            plt.imsave(save_path, gamma)

        plt.title(tile_str)

    
def viz_ndvi(tile_number: int, draw_ref: bool, save_fig: bool):
    '''
    Visualizes a Planet NDVI imagery tile.
    Red:NDVI 2016. Green:NDVI 2018. Blue: NDVI 2020.

    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - draw_ref (bool): Flag to draw reference borders on the image.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    viz_ndvi(tile_number=0, draw_ref=True, save_fig=True)
    
    Note:
    - Assumes that the Planet NDVI and reference files are in GeoTIFF format.
    - The visualization is saved in '../../../data/figures/'.
    '''

    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_ndvi = sorted(glob.glob(os.path.join(imgs_dir, '*ndvi.tif')))
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    tile_str = os.path.splitext(os.path.basename(files_ndvi[tile_number]))[0]
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_ndvi[tile_number]) as ndvi, \
         rasterio.open(files_ref[tile_number]) as reference:
        image = np.transpose(ndvi.read(), (1, 2, 0))
        image = np.clip(((image + 1) / 2), 0.75, 1)
        gamma = adjust_gamma(image, 7)
        ref = reference.read(1).astype(np.uint8)

        if draw_ref:
            edges = cv2.Canny(ref, threshold1=0, threshold2=1)
            red_mask = np.stack((edges,) * 3, axis=-1)
            gamma = np.where(red_mask > 0, (1, 0, 0), gamma)

        plt.imshow(gamma)
        plt.yticks([])
        plt.xticks([])

        if save_fig:
            save_path = os.path.join(save_dir, f"{tile_str}.png")
            plt.imsave(save_path, gamma)

        plt.title(tile_str)


def viz_s1(tile_number: int, draw_ref: bool, save_fig: bool):
    '''
    Visualizes a Sentinel-1 Band C VH imagery tile.
    Red: VH 2016. Green: VH 2018. Blue: VH 2020.

    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - draw_ref (bool): Flag to draw reference borders on the image.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    viz_s1(tile_number=0, draw_ref=True, save_fig=True)
    
    Note:
    - Assumes that the Sentinel-1 and reference files are in GeoTIFF format.
    - The visualization is saved in '../../../data/figures/'.
    '''
    
    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_s1 = sorted(glob.glob(os.path.join(imgs_dir, '*s1.tif')))
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    tile_str = os.path.splitext(os.path.basename(files_s1[tile_number]))[0]
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_s1[tile_number]) as s1, \
         rasterio.open(files_ref[tile_number]) as reference:
        image = normalize_image(np.transpose(s1.read(), (1, 2, 0)))
        image = cv2.resize(image, (400, 400))
        gamma = adjust_gamma(image, 1.2)
        ref = reference.read(1).astype(np.uint8)

        if draw_ref:
            edges = cv2.Canny(ref, threshold1=0, threshold2=1)
            red_mask = np.stack((edges,) * 3, axis=-1)
            gamma = np.where(red_mask > 0, (1, 0, 0), gamma)

        plt.imshow(gamma)
        plt.yticks([])
        plt.xticks([])

        if save_fig:
            save_path = os.path.join(save_dir, f"{tile_str}.png")
            plt.imsave(save_path, gamma)

        plt.title(tile_str)

        
def viz_palsar(tile_number: int, draw_ref: bool, save_fig: bool):
    '''
    Visualizes a ALOS/PALSAR-2 Band L HV imagery tile.
    Red: HV 2016. Green: HV 2018. Blue: HV 2020.


    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - draw_ref (bool): Flag to draw reference borders on the image.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    viz_palsar(tile_number=0, draw_ref=True, save_fig=True)
    
    Note:
    - Assumes that the palsar and reference files are in GeoTIFF format.
    - The visualization is saved in '../../../data/figures/'.
    '''
    
    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_palsar = sorted(glob.glob(os.path.join(imgs_dir, '*palsar.tif')))
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    tile_str = os.path.splitext(os.path.basename(files_palsar[tile_number]))[0]
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_palsar[tile_number]) as palsar, \
         rasterio.open(files_ref[tile_number]) as reference:
        image = normalize_image(np.transpose(palsar.read(), (1, 2, 0)))
        image = cv2.resize(image, (400, 400))
        gamma = adjust_gamma(image, 0.8)
        ref = reference.read(1).astype(np.uint8)

        if draw_ref:
            edges = cv2.Canny(ref, threshold1=0, threshold2=1)
            red_mask = np.stack((edges,) * 3, axis=-1)
            gamma = np.where(red_mask > 0, (1, 0, 0), gamma)

        plt.imshow(gamma)
        plt.yticks([])
        plt.xticks([])

        if save_fig:
            save_path = os.path.join(save_dir, f"{tile_str}.png")
            plt.imsave(save_path, gamma)

        plt.title(tile_str)


def viz_all(tile_number: int,  draw_ref: bool, save_fig: bool):
    current_dir = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(module_dir)
    imgs_dir = '../../../data/croped_data/'
    files_ref = sorted(glob.glob(imgs_dir + '*ref.tif'))
    files_planet = sorted(glob.glob(imgs_dir + '*planet.tif'))
    files_s1 = sorted(glob.glob(imgs_dir + '*s1.tif'))
    files_s2 = sorted(glob.glob(imgs_dir + '*s2.tif'))
    with rasterio.open(files_ref[tile_number]) as ref_ds,\
         rasterio.open(files_planet[tile_number]) as planet_ds,\
         rasterio.open(files_s1[tile_number]) as s1_ds,\
         rasterio.open(files_s2[tile_number]) as s2_ds:

        # Read the raster data
        ref = ref_ds.read(1).astype(np.uint8)
        nir_planet = planet_ds.read(1).astype(np.uint16)
        red_planet = planet_ds.read(2).astype(np.uint16)
        green_planet = planet_ds.read(3).astype(np.uint16)
        s1 = s1_ds.read(1)
        s2_20 = s2_ds.read(1)
        s2_21 = s2_ds.read(2)
        s2_22 = s2_ds.read(3)

        res_s1 = rescale_intensity(s1, in_range=(-30, -1),
                                   out_range=(0, 255)).astype(np.uint8)

        res_nir = rescale_intensity(nir_planet, in_range=(0, 5500),
                                    out_range=(0, 255)).astype(np.uint8)
        res_red = rescale_intensity(red_planet, in_range=(0, 4000),
                                    out_range=(0, 255)).astype(np.uint8)
        res_green = rescale_intensity(green_planet, in_range=(0, 4000),
                                      out_range=(0, 255)).astype(np.uint8)

        res_s2_20 = rescale_intensity(s2_20, in_range=(0, 4000),
                                      out_range=(0, 255)).astype(np.uint8)
        res_s2_21 = rescale_intensity(s2_21, in_range=(0, 4000),
                                      out_range=(0, 255)).astype(np.uint8)
        res_s2_22 = rescale_intensity(s2_22, in_range=(0, 4000),
                                      out_range=(0, 255)).astype(np.uint8)

        rgb = np.stack([res_nir, res_red, res_green], axis=-1)

        fig, axs = plt.subplots(2, 3, figsize=(10, 10))

        fig.suptitle('Tile ' + str(tile_number), y=0.65, size=12)

        # Plot the raster data
        axs[0, 0].imshow(ref, vmin=0, vmax=1, cmap='gray')
        axs[0, 0].set_title('Reference (5m)', pad=10)

        axs[0, 1].imshow(rgb)
        axs[0, 1].set_title('Planet Scope (5m)', pad=10)

        axs[0, 2].imshow(res_s1, cmap='gray')
        axs[0, 2].set_title('Sentinel-1 C-BAND VH (10m)', pad=10)

        axs[1, 0].imshow(res_s2_20, cmap='gray')
        axs[1, 0].set_title('Sentinel-2 Band-5 2020 (20m)', pad=10)

        axs[1, 1].imshow(res_s2_21, cmap='gray')
        axs[1, 1].set_title('Sentinel-2 Band-5 2021 (20m)', pad=10)

        axs[1, 2].imshow(res_s2_22, cmap='gray')
        axs[1, 2].set_title('Sentinel-2 Band-5 2022 (20m)', pad=10)

        plt.subplots_adjust(bottom=0, top=0.6)

    os.chdir(current_dir)
