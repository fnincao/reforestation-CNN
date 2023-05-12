'''Module to import and filter the dataset provided from Suzano'''

import geopandas as gpd


PATH_SUZANO = '/maps/fnb25/data/polygons_original/Suzano'
PATH_ATLANTIC = '/maps/fnb25/data/suport_files/Atlantica'
OUT_PATH = '/maps/fnb25/data/polygons_filtered/Suzano'

# import atlantic rainforest limits
atlantic = gpd.read_file(PATH_ATLANTIC)

# import suzano raw shapefile to gpd dataframe
suzano = gpd.read_file(PATH_SUZANO)

# convert both files to the same CRS system (ESPG: 4326)
atlantic = atlantic.to_crs(4326)

suzano = suzano.to_crs(4326)

# filter polygons that have the status = in restoration
suzano = suzano[suzano['STATUS'] == 'E']

# filter polygons that are using method = native plantion or pinus + native
suzano = suzano[(suzano['METODO'] == 'PL') | (suzano['METODO'] == 'PC')]

# filter polygons that are inside atlantic rainforest
suzano = suzano[suzano.within(atlantic.geometry.iloc[0])]

# filter polygons implementation year = 0
suzano = suzano[suzano['ANOIMPL'] != '0']

# Take only the first year from the implementation project
# when two years are available (e.g., 2015-2017)
suzano['ano'] = suzano['ANOIMPL'].str.slice(stop=4).astype(int)

# final dataset
suzano_final = suzano.loc[:, ['ano', 'geometry']].reset_index(drop=True)

suzano_final.to_file(OUT_PATH + '/suzano.shp')
