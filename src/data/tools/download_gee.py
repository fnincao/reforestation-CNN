'''Module to download image chips from GEE'''

import ee
import logging
import multiprocessing
import os
import requests
import shutil
from retry import retry


@retry(tries=10, delay=1, backoff=2)
def getResult(index, point, image, res, dir, size, sulfix):
    """
    Helper function to download image chips for a given point.

    Parameters:
    - index (int): The index of the point.
    - point (dict): The coordinates of the point.
    - image (ee.Image): The image to download chips from.
    - res (int): The desired output resolution of the chips in meters.
    - dir (str): The directory to save the downloaded chips.
    - size (int): The total number of points.
    - sulfix (str): The suffix to add to the chip filenames.
    """
    # Convert point coordinates to an Earth Engine Geometry Point
    point = ee.Geometry.Point(point['coordinates'])

    # Create a bounding region around the point for downloading the chip
    region = point.buffer(2000).bounds()

    # Generate the download URL for the chip
    url = image.getDownloadURL(
        {
            'region': region,
            'crs_transform': [res, 0, 0, 0, -res, 0],
            'crs': 'EPSG:3857',
            'format': "GEO_TIFF"
        }
    )

    ext = 'tif'

    # Send a GET request to download the chip image
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()

    # Prepare the file path and name for saving the chip image
    folder = os.path.abspath(dir)
    prefix = 'tile_'
    basename = str(index).zfill(len(str(size)))
    filename = f"{folder}/{prefix}{basename}{sulfix}.{ext}"

    # Save the chip image to the specified directory
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)


def GetImageChips(download_image: ee.Image,
                  out_resolution: int,
                  points: ee.FeatureCollection,
                  out_dir: str,
                  sulfix: str
                  ):
    """
    Function to download image chips from a Google Earth Engine image.

    Parameters:
    - download_image (ee.Image): The image from which to extract the chips.
    - out_resolution (int): The desired output resolution of the chips in meters.
    - points (ee.FeatureCollection): The points or locations where the chips will be extracted.
    - out_dir (str): The directory to save the downloaded chips.
    - suffix (str): The suffix to add to the chip filenames.

    Example Usage:
    GetImageChips(ee.Image("image_id"), 10, ee.FeatureCollection("points_collection_id"), "/path/to/save/chips", "_chip")
    """ # noqa
    # Retrieve the total number of points
    size = points.size().getInfo()

    # Configure logging
    logging.basicConfig()

    # Limit the number of points to process (5000 in this case)
    items = points.limit(5000).aggregate_array('.geo').getInfo()

    # Prepare a list of download items to process in parallel
    download_items = [(a, b, download_image,
                       out_resolution, out_dir,
                       size, sulfix) for a, b in enumerate(items)]

    # Create a multiprocessing pool with 25 workers
    pool = multiprocessing.Pool(25)

    # Download image chips in parallel using the getResult function
    pool.starmap(getResult, download_items)
    pool.close()
