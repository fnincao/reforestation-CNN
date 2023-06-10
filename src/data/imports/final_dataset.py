'''File to compile all the dataset into a final version'''
import geopandas as gpd
import glob
import pandas as pd


DATASETS_PATH = '/maps/fnb25/data/polygons_filtered'

# Create a list with all shapefiles in the folder
list_ds = glob.glob(DATASETS_PATH + '/*.gpkg')

datasets = []

# import all datasets and covert to Web Mercator
for ds in list_ds:
    temp_ds = gpd.read_file(ds).to_crs(3857)
    temp_ds['source'] = ds.split('/')[-1][:-5]
    datasets.append(temp_ds)

# Concatenate all datasets
final_dataset = pd.concat(datasets, axis=0, ignore_index=True)

# Remove rows withot year information
final_dataset = final_dataset[~final_dataset['ano'].isnull()]

# Keep only polygons that were planted until 2019
final_dataset = final_dataset[final_dataset['ano'] < 2020]

# Remove data from Pacto due to misaligment
final_dataset = final_dataset[final_dataset['source'] != 'pacto']

# Remove duplicates
final_dataset = final_dataset.drop_duplicates(subset='geometry', keep='first')

# Filter polygons bigger than one hectare
final_dataset = final_dataset[(final_dataset.geometry.area / 1000) > 1]\
                .reset_index(drop=True)

# Save final dataset
final_dataset.to_file('/maps/fnb25/data/polygons_filtered/polygons_final.gpkg',
                      fid=False)
