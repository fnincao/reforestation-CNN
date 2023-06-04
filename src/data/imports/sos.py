'''File to import/filter the dataset provided by SOS'''

import geopandas as gpd


PATH_SOS = '/maps/fnb25/data/polygons_original/SOS'
PATH_ATLANTIC = '/maps/fnb25/data/suport_files/Atlantica'

# import atlantic rainforest limits
atlantic = gpd.read_file(PATH_ATLANTIC).to_crs(4326)

# import suzano raw shapefile to gpd dataframe
sos = gpd.read_file(PATH_SOS).to_crs(4326)

# Rename the columns to be consist with the other datasets
sos = sos.rename(columns={'Ano planti': 'ano'})

# Select only the columns of interest
sos = sos.loc[:, ['ano', 'geometry']]

# filter polygons that are inside atlantic rainforest
sos = sos[sos.within(atlantic.geometry.iloc[0])]

# Save final dataset
sos.to_file('/maps/fnb25/data/polygons_filtered/sos.gpkg', fid=False)
