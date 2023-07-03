'''Module to vizualize image chips downloaded from GEE'''

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import adjust_gamma
import glob
import os
import cv2


plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def normalize_image(image):
    # Convert the image to floating-point values
    image = image.astype(np.float32)
    # Normalize the image to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def reference_image(tile_number: int, save_fig: bool):
    '''
    Visualizes a reference imagery tile.

    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    reference_image(tile_number=0, save_fig=True)

    Note:
    - Assumes that the reference files are in GeoTIFF format.

    Example Usage:
    reference_image(tile_number=0, save_fig=True)
    '''
    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    tile_str = os.path.splitext(os.path.basename(files_ref[tile_number]))[0]
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_ref[tile_number]) as reference:
        ref = reference.read(1).astype(np.uint8)
        plt.imshow(ref, vmin=0, vmax=1, cmap='gray')
        plt.yticks([])
        plt.xticks([])

        if save_fig:
            save_path = os.path.join(save_dir, f"{tile_str}.png")
            plt.imsave(save_path, ref, cmap='gray')

        plt.title(tile_str)


def planet_image(tile_number: int, draw_ref: bool, save_fig: bool):
    '''
    Visualizes a Planet RGB imagery tile.
    RGB median composite from year 2020.

    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - draw_ref (bool): Flag to draw reference borders on the image.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    planet_image(tile_number=0, draw_ref=True, save_fig=True)

    Note:
    - Assumes that the planet and reference files are in GeoTIFF format.

    Example Usage:
    planet_image(tile_number=0, draw_ref=True, save_fig=True)
    '''

    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_planet = sorted(glob.glob(os.path.join(imgs_dir, '*planet.tif')))
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    tile_str = os.path.splitext(os.path.basename(files_planet[tile_number]))[0]
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_planet[tile_number]) as planet_ds, \
         rasterio.open(files_ref[tile_number]) as reference:

        planet = np.transpose(planet_ds.read()[0:3], (1, 2, 0))
        planet = normalize_image(planet)
        planet = adjust_gamma(planet, 0.8)
        ref = reference.read(1).astype(np.uint8)

        if draw_ref:
            edges = cv2.Canny(ref, threshold1=0, threshold2=1)
            red_mask = np.stack((edges,) * 3, axis=-1)
            planet = np.where(red_mask > 0, (1, 0, 0), planet)

        plt.imshow(planet)
        plt.yticks([])
        plt.xticks([])

        if save_fig:
            save_path = os.path.join(save_dir, f"{tile_str}.png")
            plt.imsave(save_path, planet)

        plt.title(tile_str)


def ndvi_image(tile_number: int, draw_ref: bool, save_fig: bool):
    '''
    Visualizes a Planet NDVI imagery tile.
    Red:NDVI 2016. Green:NDVI 2018. Blue: NDVI 2020.

    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - draw_ref (bool): Flag to draw reference borders on the image.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    ndvi_image(tile_number=0, draw_ref=True, save_fig=True)

    Note:
    - Assumes that the Planet NDVI and reference files are in GeoTIFF format.

    Example Usage:
    ndvi_image(tile_number=0, draw_ref=True, save_fig=True)
    '''

    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_ndvi = sorted(glob.glob(os.path.join(imgs_dir, '*ndvi.tif')))
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    tile_str = os.path.splitext(os.path.basename(files_ndvi[tile_number]))[0]
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_ndvi[tile_number]) as ndvi_ds, \
         rasterio.open(files_ref[tile_number]) as reference:
        ndvi = np.transpose(ndvi_ds.read(), (1, 2, 0))
        ndvi = np.clip(((ndvi + 1) / 2), 0.75, 1)
        ndvi = adjust_gamma(ndvi, 7)
        ref = reference.read(1).astype(np.uint8)

        if draw_ref:
            edges = cv2.Canny(ref, threshold1=0, threshold2=1)
            red_mask = np.stack((edges,) * 3, axis=-1)
            ndvi = np.where(red_mask > 0, (1, 0, 0), ndvi)

        plt.imshow(ndvi)
        plt.yticks([])
        plt.xticks([])

        if save_fig:
            save_path = os.path.join(save_dir, f"{tile_str}.png")
            plt.imsave(save_path, ndvi)

        plt.title(tile_str)


def s1_image(tile_number: int, draw_ref: bool, save_fig: bool):
    '''
    Visualizes a Sentinel-1 Band C VH imagery tile.
    Red: VH 2016. Green: VH 2018. Blue: VH 2020.

    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - draw_ref (bool): Flag to draw reference borders on the image.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    s1_image(tile_number=0, draw_ref=True, save_fig=True)

    Note:
    - Assumes that the Sentinel-1 and reference files are in GeoTIFF format.

    Example Usage:
    s1_image(tile_number=0, draw_ref=True, save_fig=True)
    '''

    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_s1 = sorted(glob.glob(os.path.join(imgs_dir, '*s1.tif')))
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    tile_str = os.path.splitext(os.path.basename(files_s1[tile_number]))[0]
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_s1[tile_number]) as s1_ds, \
         rasterio.open(files_ref[tile_number]) as reference:
        s1 = normalize_image(np.transpose(s1_ds.read(), (1, 2, 0)))
        s1 = cv2.resize(s1, (400, 400))
        s1 = adjust_gamma(s1, 1.2)
        ref = reference.read(1).astype(np.uint8)

        if draw_ref:
            edges = cv2.Canny(ref, threshold1=0, threshold2=1)
            red_mask = np.stack((edges,) * 3, axis=-1)
            s1 = np.where(red_mask > 0, (1, 0, 0), s1)

        plt.imshow(s1)
        plt.yticks([])
        plt.xticks([])

        if save_fig:
            save_path = os.path.join(save_dir, f"{tile_str}.png")
            plt.imsave(save_path, s1)

        plt.title(tile_str)


def palsar_image(tile_number: int, draw_ref: bool, save_fig: bool):
    '''
    Visualizes a ALOS/PALSAR-2 Band L HV imagery tile.
    Red: HV 2016. Green: HV 2018. Blue: HV 2020.


    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - draw_ref (bool): Flag to draw reference borders on the image.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    palsar_image(tile_number=0, draw_ref=True, save_fig=True)

    Note:
    - Assumes that the palsar and reference files are in GeoTIFF format.

    Example Usage:
    palsar_image(tile_number=0, draw_ref=True, save_fig=True)
    '''

    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_palsar = sorted(glob.glob(os.path.join(imgs_dir, '*palsar.tif')))
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    tile_str = os.path.splitext(os.path.basename(files_palsar[tile_number]))[0]
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_palsar[tile_number]) as palsar_ds, \
         rasterio.open(files_ref[tile_number]) as reference:
        palsar = normalize_image(np.transpose(palsar_ds.read(), (1, 2, 0)))
        palsar = cv2.resize(palsar, (400, 400))
        palsar = adjust_gamma(palsar, 0.8)
        ref = reference.read(1).astype(np.uint8)

        if draw_ref:
            edges = cv2.Canny(ref, threshold1=0, threshold2=1)
            red_mask = np.stack((edges,) * 3, axis=-1)
            palsar = np.where(red_mask > 0, (1, 0, 0), palsar)

        plt.imshow(palsar)
        plt.yticks([])
        plt.xticks([])

        if save_fig:
            save_path = os.path.join(save_dir, f"{tile_str}.png")
            plt.imsave(save_path, palsar)

        plt.title(tile_str)


def all_images(tile_number: int,  draw_ref: bool, save_fig: bool):
    '''
    Visualizes all imagery tile.
    Planet RGB. RGB median composite from year 2020.
    Planet NDVI. Red:NDVI 2016. Green:NDVI 2018. Blue: NDVI 2020.
    Sentinel-1 C-Band. Red: VH 2016. Green: VH 2018. Blue: VH 2020.
    ALOS/PALSAR-2 L-Band. Red: HV 2016. Green: HV 2018. Blue: HV 2020.

    Parameters:
    - tile_number (int): Index of the tile to visualize.
    - draw_ref (bool): Flag to draw reference borders on the image.
    - save_fig (bool): Flag to save the visualization as a PNG image.

    Example Usage:
    all_images(tile_number=0, draw_ref=True, save_fig=True)

    Note:
    - Assumes that the images files are in GeoTIFF format.

    Example Usage:
    all_images(tile_number=0, draw_ref=True, save_fig=True)
    '''

    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/croped_data/')
    files_planet = sorted(glob.glob(os.path.join(imgs_dir, '*planet.tif')))
    files_ndvi = sorted(glob.glob(os.path.join(imgs_dir, '*ndvi.tif')))
    files_s1 = sorted(glob.glob(os.path.join(imgs_dir, '*s1.tif')))
    files_palsar = sorted(glob.glob(os.path.join(imgs_dir, '*palsar.tif')))
    files_ref = sorted(glob.glob(os.path.join(imgs_dir, '*ref.tif')))
    save_dir = os.path.join(module_dir, '../../../data/figures/')

    with rasterio.open(files_ref[tile_number]) as reference,\
         rasterio.open(files_planet[tile_number]) as planet_ds,\
         rasterio.open(files_ndvi[tile_number]) as ndvi_ds,\
         rasterio.open(files_s1[tile_number]) as s1_ds,\
         rasterio.open(files_palsar[tile_number]) as palsar_ds:

        ref = reference.read(1).astype(np.uint8)

        planet = np.transpose(planet_ds.read()[0:3], (1, 2, 0))
        planet = normalize_image(planet)
        planet = adjust_gamma(planet, 0.8)

        ndvi = np.transpose(ndvi_ds.read(), (1, 2, 0))
        ndvi = np.clip(((ndvi + 1) / 2), 0.75, 1)
        ndvi = adjust_gamma(ndvi, 7)

        s1 = normalize_image(np.transpose(s1_ds.read(), (1, 2, 0)))
        s1 = cv2.resize(s1, (400, 400))
        s1 = adjust_gamma(s1, 1.2)

        palsar = normalize_image(np.transpose(palsar_ds.read(), (1, 2, 0)))
        palsar = cv2.resize(palsar, (400, 400))
        palsar = adjust_gamma(palsar, 0.8)

        if draw_ref:
            edges = cv2.Canny(ref, threshold1=0, threshold2=1)
            red_mask = np.stack((edges,) * 3, axis=-1)
            planet = np.where(red_mask > 0, (1, 0, 0), planet)
            ndvi = np.where(red_mask > 0, (1, 0, 0), ndvi)
            s1 = np.where(red_mask > 0, (1, 0, 0), s1)
            palsar = np.where(red_mask > 0, (1, 0, 0), palsar)

        fig, axs = plt.subplots(2, 2, figsize=(6, 6))

        fig.suptitle('Tile ' + str(tile_number), size=12)

        axs[0, 0].imshow(planet)
        axs[0, 0].set_title('Planet RGB 2020 (5m)', size=9,  pad=10)
        axs[0, 0].set_yticks([])
        axs[0, 0].set_xticks([])

        axs[0, 1].imshow(ndvi)
        axs[0, 1].set_title('Temporal Planet NDVI (5m)', size=9, pad=10)
        axs[0, 1].set_yticks([])
        axs[0, 1].set_xticks([])

        axs[1, 0].imshow(s1)
        axs[1, 0].set_title('Temporal Sentinel-1 C-BAND VH (10m)', size=9, pad=10) # noqa
        axs[1, 0].set_yticks([])
        axs[1, 0].set_xticks([])

        axs[1, 1].imshow(palsar)
        axs[1, 1].set_title('Temporal ALOS/PALSAR-2 L-BAND HV (20m)', size=9, pad=10) # noqa
        axs[1, 1].set_yticks([])
        axs[1, 1].set_xticks([])

        plt.tight_layout()

        if save_fig:
            save_string = f"tile_{str(tile_number)}_all.png"
            save_path = os.path.join(save_dir, save_string)
            plt.savefig(save_path, facecolor='white')
