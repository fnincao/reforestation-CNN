'''Module to vizualize image chips downloaded from GEE'''

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity


def viz_planet(path: str):
    '''
    Helper to vizualize planet imagery download from GEE
    '''
    with rasterio.open(path) as planet:

        # Read the raster data
        nir_planet = planet.read(1).astype(np.uint16)
        red_planet = planet.read(2).astype(np.uint16)
        green_planet = planet.read(3).astype(np.uint16)

        res_nir = rescale_intensity(nir_planet, in_range=(0, 4000),
                                    out_range=(0, 255)).astype(np.uint8)
        res_red = rescale_intensity(red_planet, in_range=(0, 4000),
                                    out_range=(0, 255)).astype(np.uint8)
        res_green = rescale_intensity(green_planet, in_range=(0, 4000),
                                      out_range=(0, 255)).astype(np.uint8)

        rgb = np.stack([res_nir, res_red, res_green], axis=-1)

        plt.imshow(rgb)
        plt.title(path.split('/')[-1][:-4])


def viz_reference(path: str):
    '''
    Helper to vizualize planet imagery download from GEE
    '''
    with rasterio.open(path) as reference:

        # Read the raster data
        reference_data = reference.read(1).astype(np.uint8)

        plt.imshow(reference_data, vmin=0, vmax=1, cmap='gray')
        plt.title(path.split('/')[-1][:-4])


def viz_s1(path: str):
    '''
    Helper to vizualize Sentinel-1 imagery download from GEE
    '''
    with rasterio.open(path) as s1:

        # Read the raster data
        s1_data = s1.read(1)

        res_s1 = rescale_intensity(s1_data, in_range=(-30, -1),
                                   out_range=(0, 255)).astype(np.uint8)

        plt.imshow(res_s1,  cmap='gray')
        plt.title(path.split('/')[-1][:-4])


def viz_s2(path: str):
    '''
    Helper to vizualize Sentinel-2 imagery download from GEE
    '''
    with rasterio.open(path) as s2:

        # Read the raster data
        s2_data = s2.read(1)

        res_s2 = rescale_intensity(s2_data, in_range=(0, 4000),
                                   out_range=(0, 255)).astype(np.uint8)

        plt.imshow(res_s2,  cmap='gray')
        plt.title(path.split('/')[-1][:-4])
