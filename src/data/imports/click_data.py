'''Module to import and filter the dataset provided from CLICK'''

import geopandas as gpd
import pandas as pd


PATH_CLICK = '/maps/fnb25/data/polygons_original/SOS_new/CLICK/'
PATH_ATLANTIC = '/maps/fnb25/data/suport_files/Atlantica'

list_click = ['Click GERAL', 'Novo Click', 'click_antigo']

datasets = []

# import atlantic rainforest limits
atlantic = gpd.read_file(PATH_ATLANTIC).to_crs(4326)

for file in list_click:
    file_path = PATH_CLICK + file
    temp_file = gpd.read_file(file_path).to_crs(4326)
    datasets.append(temp_file)

click = pd.concat(datasets, axis=0, ignore_index=True)

click = click.drop_duplicates(subset='geometry', keep='first')

click = click[click['Name'].str.startswith('20') |
              click['Name'].str.endswith('21')]

# Extract the year information from the Name
click['Name'] = click['Name'].apply(lambda x: x[:4] if
                                    x.startswith('20') else x)

# Extract the year information from the Name
click['Name'] = click['Name'].apply(lambda x: x[-4:] if
                                    x[-4:-2] == '20' else x)

# Remove the columns without year information
click = click[click['Name'].apply(lambda x: len(x) == 4)]

# Rename the columns to be consist with the other datasets
click = click.rename(columns={'Name': 'ano'})

# Select only the columns of interest
click = click.loc[:, ['ano', 'geometry']]

# filter polygons that are inside atlantic rainforest
click = click[click.within(atlantic.geometry.iloc[0])].reset_index(drop=True)

# Change format of year column
click['ano'] = click['ano'].astype(int)

# Save final dataset
click.to_file('/maps/fnb25/data/polygons_filtered/click.gpkg', fid=False)
