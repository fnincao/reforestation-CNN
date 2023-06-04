'''File to import/filter the dataset provided by Suzano for the NE region'''

import json
import geopandas as gpd
from shapely.geometry import shape

PATH_SUZANO = '/maps/fnb25/data/polygons_original/Suzano'

# import suzano raw shapefile to gpd dataframe
suzano = gpd.read_file(PATH_SUZANO)

# convert files CRS system (ESPG: 31984 - SIRGAS 2000 UTM ZONE 24S)
suzano = suzano.to_crs(4326)

# filter polygons that have the status = in restoration
suzano = suzano[suzano['STATUS'] == 'E']

# filter polygons that are using method = native plantion
suzano = suzano[(suzano['METODO'] == 'PL')]

# filter polygons implementation year = 0
suzano = suzano[suzano['ANOIMPL'] != '0']

# Take only the first year from the implementation project
# when two years are available (e.g., 2015-2017)
suzano['ano'] = suzano['ANOIMPL'].str.slice(stop=4).astype(int)

# filter polygons planted till 2019, and leave only from 2010-2019
suzano = suzano[suzano['ano'] < 2020]

# create geojson str for the bbox of the ROI
geojson_str = '''
{
  "type": "Polygon",
  "coordinates": [
    [
        [-40.80626429324459,-19.866188118193687],
        [-39.05943812136959,-19.866188118193687],
        [-39.05943812136959,-17.0292149427034],
        [-40.80626429324459,-17.0292149427034],
        [-40.80626429324459,-19.866188118193687]
    ]
  ]
}
'''

# load the geojson str
geojson_obj = json.loads(geojson_str)

# create the bbox as gpd dataframe
bbox = gpd.GeoDataFrame({'geometry': [shape(geojson_obj)]},
                        crs=4326)

# define the coordinates of the bounding box
xmin, ymin, xmax, ymax = bbox.total_bounds

# filter polygons that are in the NE regions
suzano_ne = suzano.cx[xmin:xmax, ymin:ymax].reset_index(drop=True)

# Select only columns of interest
suzano_final = suzano_ne.loc[:, ['ano', 'geometry']].reset_index(drop=True)

# convert files CRS system (ESPG: 31984 - SIRGAS 2000 UTM ZONE 24S)
suzano_final = suzano_final.to_crs(31984)

# Save final dataset
suzano_final.to_file('suzano_ne.gpkg',
                     fid=False)
