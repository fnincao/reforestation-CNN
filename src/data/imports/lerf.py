'''File to import/filter the dataset provided by SOS'''

import geopandas as gpd


PATH_LERF = '/maps/fnb25/data/polygons_original/LERF'
PATH_ATLANTIC = '/maps/fnb25/data/suport_files/Atlantica'

# import atlantic rainforest limits
atlantic = gpd.read_file(PATH_ATLANTIC).to_crs(4326)

# import suzano raw shapefile to gpd dataframe
lerf = gpd.read_file(PATH_LERF).to_crs(4326)

# Change the data format
lerf['Implement'] = ('20' + lerf['Implement'].str[-2:]).astype(int)

# Rename the columns to be consist with the other datasets
lerf = lerf.rename(columns={'Implement': 'ano'})

# Select only the columns of interest
lerf = lerf.loc[:, ['ano', 'geometry']]

# filter polygons that are inside atlantic rainforest
lerf = lerf[lerf.within(atlantic.geometry.iloc[0])].reset_index(drop=True)

# Save final dataset
lerf.to_file('/maps/fnb25/data/polygons_filtered/lerf.gpkg', fid=False)
