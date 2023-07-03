"""Module to rasterize the reference polygons."""

import rasterio as rio
from rasterio.features import rasterize
from rasterio.transform import Affine
import geopandas as gpd
import warnings


def rasterize_polygon(path_file: str, pixel_res: int,
                      out_dir: str, burn_value=1):
    """
    Function to rasterize polygons to start the pre-processing for ingesting
    in the CNN.

    Parameters:
    - path_file (str): The file path to the input shapefile containing the polygons.
    - pixel_res (int): The pixel resolution in meters.
    - out_dir (str): The output directory to save the rasterized polygons.
    - burn_value (int): The value assigned to the pixels inside the polygons. Default is 1.

    Example Usage:
    rasterize_polygon('path/to/polygons.shp', 5, 'output_directory', burn_value=1)
    """ # noqa
    shp = gpd.read_file(path_file)

    # Check the proj, everything must be in metric scale
    if shp.crs.is_geographic:
        warnings.warn("This shapefile is not in metric scale,\
                       please reproject it to continue")
        raise Exception()

    shp_bounds = shp.total_bounds
    num_cols = int((shp_bounds[2] - shp_bounds[0]) / pixel_res)
    num_rows = int((shp_bounds[3] - shp_bounds[1]) / pixel_res)
    transform = Affine(pixel_res, 0, shp_bounds[0],
                       0, -pixel_res, shp_bounds[3])
    raster_name = out_dir + '/' + path_file.split('/')[-1][:-4] + '.tif'
    raster = rio.open(
        raster_name,
        'w',
        driver='GTiff',
        height=num_rows,
        width=num_cols,
        count=1,
        dtype=rio.uint8,
        transform=transform,
        crs=shp.crs,
        nodata=0,
    )

    # Rasterize geometries
    rasterize_geom = [(geom, burn_value) for geom in shp.geometry]

    raster_image = rasterize(
        shapes=rasterize_geom,
        out_shape=(num_rows, num_cols),
        transform=transform,
        fill=0,
        default_value=0,
        dtype=rio.uint8,
    )

    raster.write(raster_image, 1)
    raster.close()
