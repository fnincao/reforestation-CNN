'''Module to rename downloaded images to have str format'''

import glob
import os

def rename_chips(sensor:str):
    
    module_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_dir = os.path.join(module_dir, '../../../data/gee_data/')
    
    files = sorted(glob.glob(os.path.join(imgs_dir, f"*{sensor}.tif")))
    files_ne = sorted(glob.glob(os.path.join(imgs_dir, f"*{sensor}_ne.tif")))
    
    len_files = len(files)
    nb_len = len(files[0].split('_')[-2])
    
    for file in files_ne:
        formatted_number = str(len_files).zfill(nb_len)
        len_files +=1
        new_name = f"{imgs_dir}tile_{formatted_number}_{sensor}.tif"
        os.rename(file, new_name)