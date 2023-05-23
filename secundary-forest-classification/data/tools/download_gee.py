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
    point = ee.Geometry.Point(point['coordinates'])
    region = point.buffer(2000).bounds()

    url = image.getDownloadURL(
        {
            'region': region,
            'crs_transform': [res, 0, 0, 0, -res, 0],
            'crs': 'EPSG:3857',
            'format': "GEO_TIFF"
        }
    )
    ext = 'tif'
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()

    folder = os.path.abspath(dir)
    prefix = 'tile_'
    basename = str(index).zfill(len(str(size)))
    filename = f"{folder}/{prefix}{basename}{sulfix}.{ext}"
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", basename)


def GetImageChips(download_image: ee.Image,
                  out_resolution: int,
                  points: ee.FeatureCollection,
                  out_dir: str,
                  sulfix: str
                  ):
    '''
    Function to download image chips from a ee.Image.
    '''
    size = points.size().getInfo()
    logging.basicConfig()
    items = points.limit(5000).aggregate_array('.geo').getInfo()
    download_items = [(a, b, download_image,
                       out_resolution, out_dir,
                       size, sulfix) for a, b in enumerate(items)]
    pool = multiprocessing.Pool(25)
    pool.starmap(getResult, download_items)
    pool.close()
