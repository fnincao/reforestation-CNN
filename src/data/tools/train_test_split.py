import random
import glob
import os
import shutil

CROPED_DATA_DIR = '../../../data/croped_data/'
TRAIN_IMG_DIR = '../../../data/ai_data/train_images'
TRAIN_MASK_DIR = '../../../data/ai_data/train_masks'
VAL_IMG_DIR = '../../../data/ai_data/val_images'
VAL_MASK_DIR = '../../../data/ai_data/val_masks'


def train_test_split(train_frac: float,
                     Planet: bool,
                     Planet_red: bool,
                     S1: bool,
                     S2: bool,
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

    s2_val = [file.replace('ref.tif', 's2.tif') for file in mask_val]
    s2_train = [file.replace('ref.tif', 's2.tif') for file in mask_train]
    
    s2_val = [file.replace('ref.tif', 's2.tif') for file in mask_val]
    s2_train = [file.replace('ref.tif', 's2.tif') for file in mask_train]

    train_images = (planet_train * Planet) + (red_train * Planet_red) +\
                   (s1_train * S1) + (s2_train * S2)
        
    val_images = (planet_val * Planet) + (red_val * Planet_red) +\
                 (s1_val * S1) + (s2_val * S2)

    for image in train_images:
        shutil.copy(image, TRAIN_IMG_DIR)

    for image in mask_train:
        shutil.copy(image, TRAIN_MASK_DIR)

    for image in val_images:
        shutil.copy(image, VAL_IMG_DIR)

    for image in mask_val:
        shutil.copy(image, VAL_MASK_DIR)

    os.chdir(current_dir)
