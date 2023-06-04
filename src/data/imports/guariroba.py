'''File to import/filter the dataset provided by Guariroba'''

import geopandas as gpd
import glob
import pandas as pd
import numpy as np


GUARIROBA_PATH = '/maps/fnb25/data/polygons_original/Guariroba'
PATH_ATLANTIC = '/maps/fnb25/data/suport_files/Atlantica'

# import atlantic rainforest limits
atlantic = gpd.read_file(PATH_ATLANTIC).to_crs(4326)

# Create a list with all shapefiles in the folder
list_shp = glob.glob(GUARIROBA_PATH + '/*.shp')

# Remove the shapefile from pinus plantation
list_shp.pop(5)

datasets = []

# Load shapefiles datasets
for shp in list_shp:
    temp_file = gpd.read_file(shp)
    datasets.append(temp_file)

# Concatenate all dataset into one dataframe
guariroba = pd.concat(datasets, axis=0, ignore_index=True)

# Create a column year
guariroba['ano'] = np.nan

# Select only columns of interest
guariroba = guariroba.loc[:, ['ano', 'geometry']]

# filter polygons that are inside atlantic rainforest
guariroba = guariroba[guariroba.within(atlantic.geometry.iloc[0])]

# Define the CRS
guariroba = guariroba.to_crs(4326)

# Save final dataset
guariroba.to_file('/maps/fnb25/data/polygons_filtered/guariroba.gpkg',
                  fid=False)
