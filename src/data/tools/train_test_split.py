import random
import glob
import os
import shutil

CROPED_DATA_DIR = '../../../data/croped_data/'
TRAIN_IMG_DIR = '../../../data/ai_data/train_images'
TRAIN_MASK_DIR = '../../../data/ai_data/train_masks'
VAL_IMG_DIR = '../../../data/ai_data/val_images'
VAL_MASK_DIR = '../../../data/ai_data/val_masks'


def train_test_split(train_frac: float):

    current_dir = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(module_dir)

    files = sorted(glob.glob(CROPED_DATA_DIR + '*ref.tif'))
    mask_val = sorted(random.sample(files, int(len(files) * train_frac)))
    mask_train = sorted([file for file in files if file not in mask_val])

    planet_val = [file.replace('ref.tif', 'planet.tif') for file in mask_val]
    planet_train = [file.replace('ref.tif',
                                 'planet.tif') for file in mask_train]

    s1_val = [file.replace('ref.tif', 's1.tif') for file in mask_val]
    s1_train = [file.replace('ref.tif', 's1.tif') for file in mask_train]

    s2_val = [file.replace('ref.tif', 's2.tif') for file in mask_val]
    s2_train = [file.replace('ref.tif', 's2.tif') for file in mask_train]

    train_images = planet_train + s1_train + s2_train
    val_images = planet_val + s1_val + s2_val

    for image in train_images:
        shutil.copy(image, TRAIN_IMG_DIR)

    for image in mask_train:
        shutil.copy(image, TRAIN_MASK_DIR)

    for image in val_images:
        shutil.copy(image, VAL_IMG_DIR)

    for image in mask_val:
        shutil.copy(image, VAL_MASK_DIR)

    os.chdir(current_dir)
