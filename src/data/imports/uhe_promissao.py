'''Module to import and filter the dataset provided from UHE PROMISSAO'''
import glob
import geopandas as gpd
import fiona
import zipfile
import os
import numpy as np

PATH_ATLANTIC = '/maps/fnb25/data/suport_files/Atlantica'
PATH_UHE = '/maps/fnb25/data/polygons_original/SOS_new/UHE PROMISS/'

# Load atlantic rainforest boundaries
atlantic = gpd.read_file(PATH_ATLANTIC).to_crs(4326)

# Enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['kml'] = 'rw'

# Enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw'

# Load the files of the BD in KMZ
kmz_files = glob.glob(PATH_UHE + '*.kmz')

geometry = []

for file in kmz_files:
    # Decompress KMZ into KML files
    with zipfile.ZipFile(file, 'r') as z:
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

# Load the files of the BD in KML
kml_files = glob.glob(PATH_UHE + '*.kml')

for file in kml_files:
    temp_file = gpd.read_file(file, driver='KML')
    # Transform multipolygons into single polygons per feature
    exp_temfile = temp_file.explode(index_parts=False)
    # Check if the data is valid and correct those which aren't
    for value in exp_temfile.is_valid:
        if not value:
            exp_temfile['geometry'] = exp_temfile.geometry.buffer(0)
    for item in exp_temfile.geometry.tolist():
        geometry.append(item)


# Load the data info a geopandas dataframe
uhe_promissao = gpd.GeoDataFrame({'ano': np.nan, 'geometry': geometry})

# Define the CRS
uhe_promissao.crs = 'epsg:4326'

# Filter point data
uhe_promissao = uhe_promissao[uhe_promissao.geom_type != 'Point']

uhe_promissao = uhe_promissao[uhe_promissao.within(atlantic.geometry.iloc[0])]\
                .reset_index(drop=True)

uhe_promissao.to_file('/maps/fnb25/data/polygons_filtered/uhe_promissao.gpkg',
                      fid=False)
