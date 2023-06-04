'''Module to extract sampling points for creating the image chips'''
import geopandas as gpd
from shapely.geometry import Polygon


def create_sampling_points(ds_path: str, spatial_res: int,
                           out_path: str, epsg='EPSG:3857',
                           inter_area=10000, save_grid=False):
    """
    Create sampling points to extract image chips for use in AI frameworks.

    Parameters:
    - ds_path (str): Path to the dataset.
    - spatial_res (int): Spatial resolution in meters.
    - out_path (str): Path to save the sampling points.
    - epsg (str): Coordinate reference system. Must be metric. Default: EPSG:3857. 
    - inter_area (int): Intersection area in square meters. Default: 10000.
    - save_grid (bool): Flag to save the grid in geopackage format. Default: False.

    Returns:
    - centroids (geopackage): Centroids of cells with an
    intersection area greater than 10000m2, in geopackage format.

    Example Usage:
    create_sampling_points(ds_path='user/Downloads', spatial_res=2000,
    out_path='user/Documents', epsg='EPSG:4326', inter_area=5000, save_grid=True)
    """ # noqa

    ds = gpd.read_file(ds_path).to_crs(epsg)
    # Check the proj, everything must be in metric scale
    # If not convert to EPSG:3857
    if ds.crs.is_geographic:
        ds.to_crs(3857)
        print('Warning: Converted the Dataset into EPSG:3857')

    # extract the bbox form the dataset
    bbox = ds.total_bounds.tolist()

    # create a polygon to represent the dataset
    geometry = [Polygon([(bbox[0], bbox[1]),
                         (bbox[2], bbox[1]),
                         (bbox[2], bbox[3]),
                         (bbox[0], bbox[3])])]

    roi = gpd.GeoDataFrame(geometry=geometry, crs=epsg)

    # calculate the number of rows and columns
    num_rows = int((bbox[3] - bbox[1]) / spatial_res) + 1
    num_cols = int((bbox[2] - bbox[0]) / spatial_res) + 1

    # generate the grid cells
    grid_cells = []
    for row in range(num_rows):
        for col in range(num_cols):
            minx = bbox[0] + col * spatial_res
            maxx = minx + spatial_res
            miny = bbox[1] + row * spatial_res
            maxy = miny + spatial_res
            polygon = Polygon([(minx, miny), (maxx, miny),
                               (maxx, maxy), (minx, maxy)])
            grid_cells.append(polygon)

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=roi.crs)

    # filter the grid just for the intersections with the ds
    inter = gpd.sjoin(grid, ds, how='inner', op='intersects')

    # check if there are duplicates
    duplicates = inter.duplicated(subset='geometry')

    deduplicated = inter[~duplicates].reset_index(drop=True)

    # Create an empty GeoDataFrame to store the filtered polygons
    polygons_to_keep = []

    # select only the polygons that have an intesection area > inter_area
    for idx, row in deduplicated.iterrows():
        geometry = row['geometry']
        intersection_area = ds.intersection(geometry).area
        if intersection_area.max() >= inter_area:
            polygons_to_keep.append(geometry)

    final_grid = gpd.GeoDataFrame(geometry=polygons_to_keep,
                                  crs=roi.crs)

    # calculates the centroids from the created grids
    final_grid['centroid'] = final_grid['geometry'].centroid

    final_grid.centroid.to_file(out_path + '/points.gpkg')

    if save_grid:
        final_grid.geometry.to_file(out_path + '/sampling_grid.gpkg')
