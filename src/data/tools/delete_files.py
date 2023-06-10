'''Module to delete files from data folder'''

import os

# Specify the directory path
directory ='../../../data/ai_data/saved_images'

# Iterate over the files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # Check if the path is a file and matches the pattern
    if os.path.isfile(file_path) and filename.endswith('.png'):
        # Remove the file
        os.remove(file_path)
        
        
# Specify the directory path
directories = ['../../../data/gee_data',
             '../../../data/croped_data', 
             '../../../data/ai_data/train_images',
             '../../../data/ai_data/val_masks',
             '../../../data/ai_data/train_masks',
             '../../../data/ai_data/val_images']

for directory in directories:
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the path is a file and matches the pattern
        if os.path.isfile(file_path) and filename.endswith('.tif'):
            # Remove the file
            os.remove(file_path)
