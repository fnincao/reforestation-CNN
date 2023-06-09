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
                     Planet_red: bool,
                     S1: bool,
                     NDVI: bool,
                     Palsar: bool):

    current_dir = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(module_dir)

    files = sorted(glob.glob(CROPED_DATA_DIR + '*ref.tif'))
    mask_val = sorted(random.sample(files, int(len(files) * train_frac)))
    mask_train = sorted([file for file in files if file not in mask_val])

    planet_val = [file.replace('ref.tif', 'planet.tif') for file in mask_val]
    planet_train = [file.replace('ref.tif',
                                 'planet.tif') for file in mask_train]
    
    red_val = [file.replace('ref.tif', 'red.tif') for file in mask_val]
    red_train = [file.replace('ref.tif','red.tif') for file in mask_train]

    s1_val = [file.replace('ref.tif', 's1.tif') for file in mask_val]
    s1_train = [file.replace('ref.tif', 's1.tif') for file in mask_train]

    NDVI_val = [file.replace('ref.tif', 'ndvi.tif') for file in mask_val]
    NDVI_train = [file.replace('ref.tif', 'ndvi.tif') for file in mask_train]
    
    palsar_val = [file.replace('ref.tif', 'palsar.tif') for file in mask_val]
    palsar_train = [file.replace('ref.tif', 'palsar.tif') for file in mask_train]

    train_images = (planet_train * Planet) + (red_train * Planet_red) +\
                   (s1_train * S1) + (NDVI_train * NDVI) + (palsar_train * Palsar)
        
    val_images = (planet_val * Planet) + (red_val * Planet_red) +\
                 (s1_val * S1) + (NDVI_val * NDVI) + (palsar_val * Palsar)

    for image in train_images:
        shutil.copy(image, TRAIN_IMG_DIR)

    for image in mask_train:
        shutil.copy(image, TRAIN_MASK_DIR)

    for image in val_images:
        shutil.copy(image, VAL_IMG_DIR)

    for image in mask_val:
        shutil.copy(image, VAL_MASK_DIR)

    os.chdir(current_dir)
