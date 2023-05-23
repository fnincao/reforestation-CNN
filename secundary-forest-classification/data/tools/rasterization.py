""" Module to rasterize the reference polygons. """

import rasterio as rio
from rasterio.features import rasterize
from rasterio.transform import Affine
import geopandas as gpd
import warnings

path_shapefile = '/maps/fnb25/data/polygons_filtered/polygons.gpkg'
out_directory = '/maps/fnb25'


def rasterize_polygon(path_shp: str, pixel_res: int,
                      out_dir: str, burn_value=1):
    """
    Function to rasterize polygons to start the pre-processing for ingesting
    in the CNN. The polygon shapefile MUST be in a projected coordinate system.
    Pixel resolution must be provided in meters. Burn value is the number that
    will be assign to the pixels inside the polygons, default = 1.
    """
    shp = gpd.read_file(path_shp)

    # Check the proj, everything must be in metric scale
    if shp.crs.is_geographic:
        warnings.warn("This shapefile is not in metrics scale, \
                      please reproject it to continue")
        raise Exception()

    shp_bounds = shp.total_bounds
    num_cols = int((shp_bounds[2] - shp_bounds[0]) / pixel_res)
    num_rows = int((shp_bounds[3] - shp_bounds[1]) / pixel_res)
    transform = Affine(pixel_res, 0, shp_bounds[0],
                       0, -pixel_res, shp_bounds[3])
    raster_name = out_directory + '/' + path_shp.split('/')[-1][:-4] + '.tif'
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
