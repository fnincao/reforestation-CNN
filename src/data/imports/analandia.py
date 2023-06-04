'''File to import/filter the dataset provided by Analandia'''

import geopandas as gpd
import zipfile
import os
import fiona
import numpy as np


ANALANDIA_PATH = '/maps/fnb25/data/polygons_original/SOS_new/SOS_MA_Analandia.kmz' # noqa E501
PATH_ATLANTIC = '/maps/fnb25/data/suport_files/Atlantica'

# import atlantic rainforest limits
atlantic = gpd.read_file(PATH_ATLANTIC).to_crs(4326)

# Enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['kml'] = 'rw'

# Enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw'

geometry = []

# Decompress KMZ into KML file
with zipfile.ZipFile(ANALANDIA_PATH, 'r') as z:
    z.extractall('/maps/fnb25/')
temp_file = gpd.read_file('/maps/fnb25/doc.kml', driver='KML')
# Transform multipolygons into single polygons per feature
exp_temfile = temp_file.explode(index_parts=False)
# Check if the data is valid and correct those which aren't
for value in exp_temfile.is_valid:
    if not value:
        exp_temfile['geometry'] = exp_temfile.geometry.buffer(0)
for item in exp_temfile.geometry.tolist():
    geometry.append(item)
os.remove('/maps/fnb25/doc.kml')

# Create geopandas dataframe
analandia = gpd.GeoDataFrame({'ano': np.nan, 'geometry': geometry})

# Define the CRS
analandia.crs = 'epsg:4326'

# filter polygons that are inside atlantic rainforest
analandia = analandia[analandia.within(atlantic.geometry.iloc[0])]

# Save final dataset
analandia.to_file('/maps/fnb25/data/polygons_filtered/analandia.gpkg',
                  fid=False)
