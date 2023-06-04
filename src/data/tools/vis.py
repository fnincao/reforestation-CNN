'''Module to vizualize image chips downloaded from GEE'''

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
import glob
import os

plt.rcParams['figure.dpi'] = 300


def viz_reference(tile_number: int):
    '''
    Helper to vizualize planet imagery download from GEE
    '''
    current_dir = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(module_dir)
    imgs_dir = '../../../data/croped_data/'
    files_ref = sorted(glob.glob(imgs_dir + '*ref.tif'))
    with rasterio.open(files_ref[tile_number]) as reference:

        # Read the raster data
        reference_data = reference.read(1).astype(np.uint8)

        plt.imshow(reference_data, vmin=0, vmax=1, cmap='gray')
        plt.title(files_ref[tile_number].split('/')[-1][:-4])

    os.chdir(current_dir)


def viz_planet(tile_number: int):
    '''
    Helper to vizualize planet imagery download from GEE
    '''
    current_dir = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(module_dir)
    imgs_dir = '../../../data/croped_data/'
    files_planet = sorted(glob.glob(imgs_dir + '*planet.tif'))
    with rasterio.open(files_planet[tile_number]) as planet:

        # Read the raster data
        nir_planet = planet.read(1).astype(np.uint16)
        red_planet = planet.read(2).astype(np.uint16)
        green_planet = planet.read(3).astype(np.uint16)

        res_nir = rescale_intensity(nir_planet, in_range=(0, 5500),
                                    out_range=(0, 255)).astype(np.uint8)
        res_red = rescale_intensity(red_planet, in_range=(0, 4000),
                                    out_range=(0, 255)).astype(np.uint8)
        res_green = rescale_intensity(green_planet, in_range=(0, 4000),
                                      out_range=(0, 255)).astype(np.uint8)

        rgb = np.stack([res_nir, res_red, res_green], axis=-1)

        plt.imshow(rgb)
        plt.title(files_planet[tile_number].split('/')[-1][:-4])

    os.chdir(current_dir)


def viz_s1(tile_number: int):
    '''
    Helper to vizualize Sentinel-1 imagery download from GEE
    '''
    current_dir = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(module_dir)
    imgs_dir = '../../../data/croped_data/'
    files_s1 = sorted(glob.glob(imgs_dir + '*s1.tif'))
    with rasterio.open(files_s1[tile_number]) as s1:

        # Read the raster data
        s1_data = s1.read(1)

        res_s1 = rescale_intensity(s1_data, in_range=(-30, -1),
                                   out_range=(0, 255)).astype(np.uint8)

        plt.imshow(res_s1,  cmap='gray')
        plt.title(files_s1[tile_number].split('/')[-1][:-4])

    os.chdir(current_dir)


def viz_s2(tile_number: int):
    '''
    Helper to vizualize Sentinel-2 imagery download from GEE
    '''
    current_dir = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(module_dir)
    imgs_dir = '../../../data/croped_data/'
    files_s2 = sorted(glob.glob(imgs_dir + '*s2.tif'))
    with rasterio.open(files_s2[tile_number]) as s2_ds:

        # Read the raster data
        s2_20 = s2_ds.read(1)
        s2_21 = s2_ds.read(2)
        s2_22 = s2_ds.read(3)

        res_s2_20 = rescale_intensity(s2_20, in_range=(0, 4000),
                                      out_range=(0, 255)).astype(np.uint8)
        res_s2_21 = rescale_intensity(s2_21, in_range=(0, 4000),
                                      out_range=(0, 255)).astype(np.uint8)
        res_s2_22 = rescale_intensity(s2_22, in_range=(0, 4000),
                                      out_range=(0, 255)).astype(np.uint8)

        fig, axs = plt.subplots(1, 3, figsize=(10, 10))
        fig.suptitle(files_s2[tile_number].split('/')[-1][:-4], y=0.7)

        axs[0].imshow(res_s2_20, cmap='gray')
        axs[0].set_title('Sentinel-2 Band-5 2020 (20m)', pad=10)

        axs[1].imshow(res_s2_21, cmap='gray')
        axs[1].set_title('Sentinel-2 Band-5 2021 (20m)', pad=10)

        axs[2].imshow(res_s2_22, cmap='gray')
        axs[2].set_title('Sentinel-2 Band-5 2022 (20m)', pad=10)

    os.chdir(current_dir)


def viz_all(tile_number: int):
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
