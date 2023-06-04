'''File to import/filter the dataset provided by Pacto'''

import geopandas as gpd
import pandas as pd


PATH_PACTO = '/maps/fnb25/data/polygons_original/Pacto'
PATH_ATLANTIC = '/maps/fnb25/data/suport_files/Atlantica'

# import atlantic rainforest limits
atlantic = gpd.read_file(PATH_ATLANTIC)

# import suzano raw shapefile to gpd dataframe
pacto = gpd.read_file(PATH_PACTO)

# convert both files to the same CRS system (ESPG: 4326)
atlantic = atlantic.to_crs(4326)

pacto = pacto.to_crs(4326)

# filter polygons that have the are provided by Suzano
pacto = pacto[pacto['responsave'] != 'Suzano Papel e Celulose']

# filter polygons that have poor information
pacto = pacto[pacto['responsave'].notna()]

# filter polygons that have the are provided by SOS
pacto = pacto[pacto['responsave'] != 'SOS']

# filter polygons that are inside atlantic rainforest
pacto = pacto[pacto.within(atlantic.geometry.iloc[0])]

# Take only the year from the implementation project
pacto['ano'] = pd.to_datetime(pacto['data_criac']).dt.year

# Select only columns of interest
pacto_final = pacto.loc[:, ['ano', 'geometry']].reset_index(drop=True)

# Save final dataset
pacto_final.to_file('/maps/fnb25/data/polygons_filtered/pacto.gpkg', fid=False)
