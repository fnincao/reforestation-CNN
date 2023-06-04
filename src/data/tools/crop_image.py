'''
Module to crop image chips to have the same spatial extext
and the right pixel resolution
'''
import glob
import rasterio
from rasterio.windows import Window
from rasterio.windows import from_bounds


def crop_ref_img(path: str, out_dir: str):
    """
    Function to crop the reference raster to the right size for
    the U-NET (400x400). The raster will be croped using the central
    pixel as a reference.
    """
    # Open the raster image file

    files = sorted(glob.glob(path + '/*ref.tif'))

    for file in files:
        with rasterio.open(file) as src:
            # Get the image dimensions
            width = src.width
            height = src.height

            # Calculate the central coordinates
            center_x = width // 2
            center_y = height // 2

            # Calculate the window coordinates
            half_width = 200  # Half of the desired window width
            left = center_x - half_width
            top = center_y - half_width
            right = center_x + half_width
            bottom = center_y + half_width

            # Create the window
            window = Window.from_slices((top, bottom), (left, right))

            # Read the windowed data
            windowed_data = src.read(window=window)

            # Update the metadata for the windowed data
            window_transform = rasterio.windows.transform(window,
                                                          src.transform)
            window_profile = src.profile.copy()
            window_profile.update({
                'height': window.height,
                'width': window.width,
                'transform': window_transform
            })

        # Write the windowed data to a new raster file

        save_string = out_dir + '/' + file.split('/')[-1]
        with rasterio.open(save_string, 'w', **window_profile) as dst:
            dst.write(windowed_data)


def crop_other_img(sensor: str, to_crop_path: str,
                   out_dir: str, ref_path: str):
    '''
    Function to crop all the other raster to the right size for
    U-NET. The raster will be croped based on the spatial extent
    of the reference raster.
    '''
    ref_files = sorted(glob.glob(ref_path + '/*ref.tif'))

    if sensor == 'planet':
        to_crop_files = sorted(glob.glob(to_crop_path + '/*planet.tif'))

    elif sensor == 's1':
        to_crop_files = sorted(glob.glob(to_crop_path + '/*s1.tif'))

    elif sensor == 's2':
        to_crop_files = sorted(glob.glob(to_crop_path + '/*s2.tif'))

    for ref_file, to_crop_file in zip(ref_files, to_crop_files):
        with rasterio.open(ref_file) as src:
            xmin, ymin, xmax, ymax = src.bounds

        with rasterio.open(to_crop_file) as data:
            # Create a window from the bounding box coordinates
            window = from_bounds(xmin, ymin, xmax, ymax,
                                 transform=data.transform)

            # Read the cropped raster data within the window
            cropped_data = data.read(window=window)

            # Create a new transform for the cropped raster
            cropped_transform = data.window_transform(window)

            # Create a new profile for the cropped raster
            cropped_profile = data.profile
            cropped_profile.update({
                'height': window.height,
                'width': window.width,
                'transform': cropped_transform
            })
            save_string = out_dir + '/' + to_crop_file.split('/')[-1]
            # Write the cropped raster to a new file
            with rasterio.open(save_string, 'w', **cropped_profile) as dst:
                dst.write(cropped_data)
