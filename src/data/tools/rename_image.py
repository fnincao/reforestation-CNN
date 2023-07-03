'''Module to rename downloaded images to have str format'''

import glob
import os


def rename_chips(sensor: str):
    """
    Function to rename downloaded images with a standardized naming format.

    Parameters:
    - sensor (str): The sensor name or identifier.

    Example Usage:
    rename_chips(sensor:'s1')
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/gee_data/')

    # Retrieve a list of downloaded images matching the sensor pattern
    files = sorted(glob.glob(os.path.join(imgs_dir, f"*{sensor}.tif")))
    files_ne = sorted(glob.glob(os.path.join(imgs_dir, f"*{sensor}_ne.tif")))

    # Determine the total number of files and the number of
    # digits required for numbering
    len_files = len(files)
    nb_len = len(files[0].split('_')[-2])

    # Rename the files in a sequential format
    for file in files_ne:
        formatted_number = str(len_files).zfill(nb_len)
        len_files += 1
        new_name = f"{imgs_dir}tile_{formatted_number}_{sensor}.tif"
        os.rename(file, new_name)
