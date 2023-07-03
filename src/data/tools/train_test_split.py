"""Module for train-test split for AI model training."""

import random
import glob
import os
import shutil

CROPED_DATA_DIR = '../../../data/croped_data/'
TRAIN_IMG_DIR = '../../../data/ai_data/train_images'
TRAIN_MASK_DIR = '../../../data/ai_data/train_masks'
VAL_IMG_DIR = '../../../data/ai_data/val_images'
VAL_MASK_DIR = '../../../data/ai_data/val_masks'

random.seed(42)


def train_test_split(train_frac: float,
                     Planet: bool,
                     S1: bool,
                     NDVI: bool,
                     Palsar: bool):
    """
    Split the dataset into train and validation sets based on the given
    fractions and data sources.
    Copy the corresponding images and masks to the appropriate directories.

    Parameters:
    - train_frac (float): The fraction of data to be used for training.
    - Planet (bool): Flag indicating whether to include Planet data.
    - S1 (bool): Flag indicating whether to include Sentinel-1 data.
    - NDVI (bool): Flag indicating whether to include NDVI data.
    - Palsar (bool): Flag indicating whether to include PALSAR data.

    Example Usage:
    train_test_split(train_frac=0.8, Planet=True, S1=True, NDVI=True, Palsar=True)
    """ # noqa

    current_dir = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(module_dir)

    # Get a list of all reference raster files
    files = sorted(glob.glob(CROPED_DATA_DIR + '*ref.tif'))

    # Split the files into train and validation sets
    mask_val = sorted(random.sample(files, int(len(files) * train_frac)))
    mask_train = sorted([file for file in files if file not in mask_val])

    # Generate file paths for each data source and set
    planet_val = [file.replace('ref.tif', 'planet.tif') for file in mask_val]
    planet_train = [file.replace('ref.tif',
                                 'planet.tif') for file in mask_train]

    s1_val = [file.replace('ref.tif', 's1.tif') for file in mask_val]
    s1_train = [file.replace('ref.tif',
                             's1.tif') for file in mask_train]

    NDVI_val = [file.replace('ref.tif', 'ndvi.tif') for file in mask_val]
    NDVI_train = [file.replace('ref.tif',
                               'ndvi.tif') for file in mask_train]

    palsar_val = [file.replace('ref.tif', 'palsar.tif') for file in mask_val]
    palsar_train = [file.replace('ref.tif',
                                 'palsar.tif') for file in mask_train]

    # Concatenate the data sources based on the provided flags
    train_images = (planet_train * Planet) + \
                   (s1_train * S1) + (NDVI_train * NDVI) + \
                   (palsar_train * Palsar)

    val_images = (planet_val * Planet) + \
                 (s1_val * S1) + (NDVI_val * NDVI) + (palsar_val * Palsar)

    # Copy the images to the appropriate directories
    for image in train_images:
        shutil.copy(image, TRAIN_IMG_DIR)

    for image in mask_train:
        shutil.copy(image, TRAIN_MASK_DIR)

    for image in val_images:
        shutil.copy(image, VAL_IMG_DIR)

    for image in mask_val:
        shutil.copy(image, VAL_MASK_DIR)

    os.chdir(current_dir)
