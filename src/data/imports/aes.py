'''File to import/filter the dataset provided by AES BRASIL'''

import geopandas as gpd
import glob
import pandas as pd


AES_PATH = '/maps/fnb25/data/polygons_original/AES_Brasil'
PATH_ATLANTIC = '/maps/fnb25/data/suport_files/Atlantica'

# import atlantic rainforest limits
atlantic = gpd.read_file(PATH_ATLANTIC).to_crs(4326)

# Create a list with all shapefiles in the folder
list_shp = glob.glob(AES_PATH + '/*.shp')


# Select columns UHE BARIRI
bariri = gpd.read_file(list_shp[0]).loc[:, ['ATUAL_SIT',
                                            'ANO_PLANT',
                                            'geometry']].to_crs(4326)

# Filter based on platation year and current situation
bariri = bariri[(bariri['ANO_PLANT'] != 0) &
                ((bariri['ATUAL_SIT'] == 'EM RESTAURAÇÃO') |
                 (bariri['ATUAL_SIT'] == 'RESTAURADA'))].reset_index(drop=True)

# Change the name to be the same in all datasets
bariri = bariri.rename(columns={'ATUAL_SIT': 'SIT', 'ANO_PLANT': 'ANO'})

# Select columns UHE Limoeiro
limoeiro = gpd.read_file(list_shp[1]).loc[:, ['SITUACAO',
                                              'ANO_PLANTI',
                                              'geometry']].to_crs(4326)

# Filter based on platation year and current situation
limoeiro = limoeiro[(limoeiro['ANO_PLANTI'] > 0) &
                    (limoeiro['SITUACAO'] == 'RESTAURADA')] \
                    .reset_index(drop=True)

# Change the name to be the same in all datasets
limoeiro = limoeiro.rename(columns={'SITUACAO': 'SIT', 'ANO_PLANTI': 'ANO'})

# Select columns UHE Caconde
caconde = gpd.read_file(list_shp[2]).loc[:, ['SITUACAO',
                                             'ANO_PLANTI',
                                             'geometry']].to_crs(4326)

# Filter based on platation year and current situation
caconde = caconde[(caconde['ANO_PLANTI'].astype(float) > 0) &
                  ((caconde['SITUACAO'] == 'EM RESTAURAÇÃO') |
                   (caconde['SITUACAO'] == 'RESTAURADA'))] \
                   .reset_index(drop=True)

caconde['ANO_PLANTI'] = caconde['ANO_PLANTI'].astype(int)

# Change the name to be the same in all datasets
caconde = caconde.rename(columns={'SITUACAO': 'SIT', 'ANO_PLANTI': 'ANO'})

# Select columns UHE Promissao
promissao = gpd.read_file(list_shp[3]).loc[:, ['ATUAL_SITU',
                                               'ANO_PLANTI',
                                               'geometry']].to_crs(4326)

# Filter based on platation year and current situation
promissao = promissao[((promissao['ANO_PLANTI'].astype(float) > 0) &
                      ((promissao['ANO_PLANTI'].astype(float) < 2030))) &
                      ((promissao['ATUAL_SITU'] == 'EM RESTAURAÇÃO') |
                      (promissao['ATUAL_SITU'] == 'RESTAURADA'))] \
                      .reset_index(drop=True)

promissao['ANO_PLANTI'] = promissao['ANO_PLANTI'].astype(int)

# Change the name to be the same in all datasets
promissao = promissao.rename(columns={'ATUAL_SITU': 'SIT',
                                      'ANO_PLANTI': 'ANO'})

# Select columns PCH Mogi
mogi = gpd.read_file(list_shp[4]).loc[:, ['SIT_ATUAL',
                                          'ANO_PLANTI',
                                          'geometry']].to_crs(4326)

# Filter based on platation year and current situation
mogi = mogi[(mogi['ANO_PLANTI'].astype(float) > 0) &
            ((mogi['SIT_ATUAL'] == 'EM RESTAURAÇÃO') |
             (mogi['SIT_ATUAL'] == 'RESTAURADA'))].reset_index(drop=True)

mogi['ANO_PLANTI'] = mogi['ANO_PLANTI'].astype(int)

# Change the name to be the same in all datasets
mogi = mogi.rename(columns={'SIT_ATUAL': 'SIT', 'ANO_PLANTI': 'ANO'})

# Select columns PCH BARRA BONITA
barra = gpd.read_file(list_shp[5]).loc[:, ['ATUAL_SITU',
                                           'ANO_PLANT',
                                           'geometry']].to_crs(4326)

# Filter based on platation year and current situation
barra = barra[(barra['ANO_PLANT'] > 0) &
              ((barra['ATUAL_SITU'] == 'EM RESTAURAÇÃO') |
               (barra['ATUAL_SITU'] == 'RESTAURADA'))].reset_index(drop=True)

# Change the name to be the same in all datasets
barra = barra.rename(columns={'ATUAL_SITU': 'SIT', 'ANO_PLANT': 'ANO'})

# Select columns UHE IBITINGA
ibitinga = gpd.read_file(list_shp[6]).loc[:, ['ATUAL_SITU',
                                              'ANO_PLANTI',
                                              'geometry']].to_crs(4326)

# Replace the values with 2 years and just let the first one (2015/2019)
ibitinga['ANO_PLANTI'] = ibitinga['ANO_PLANTI'].str.slice(stop=4).astype(float)

# Filter based on platation year and current situation
ibitinga = ibitinga[(ibitinga['ANO_PLANTI'] > 0) &
                    ((ibitinga['ATUAL_SITU'] == 'EM RESTAURAÇÃO') |
                     (ibitinga['ATUAL_SITU'] == 'RESTAURADA'))] \
                     .reset_index(drop=True)

# Change the name to be the same in all datasets
ibitinga = ibitinga.rename(columns={'ATUAL_SITU': 'SIT', 'ANO_PLANTI': 'ANO'})

# Select columns UHE NOVA AVANHANDAVA
nova = gpd.read_file(list_shp[7]).loc[:, ['SITUACAO',
                                          'ANO_PLANTI',
                                          'geometry']].to_crs(4326)

# Filter based on platation year and current situation
nova = nova[(nova['ANO_PLANTI'] > 0) &
            ((nova['SITUACAO'] == 'EM RESTAURAÇÃO') |
             (nova['SITUACAO'] == 'RESTAURADA'))].reset_index(drop=True)

# Change the name to be the same in all datasets
nova = nova.rename(columns={'SITUACAO': 'SIT', 'ANO_PLANTI': 'ANO'})

# Select columns UHE AGUA VERMELHA
agua = gpd.read_file(list_shp[8]).loc[:, ['SIT_ATUAL',
                                          'ANO_PLANTI',
                                          'geometry']].to_crs(4326)

agua['ANO_PLANTI'] = agua['ANO_PLANTI'].str.slice(stop=4).astype(float)

# Filter based on platation year and current situation
agua = agua[(agua['ANO_PLANTI'] > 0) &
            ((agua['SIT_ATUAL'] == 'EM RESTAURAÇÃO') |
             (agua['SIT_ATUAL'] == 'RESTAURADA'))].reset_index(drop=True)

# Change the name to be the same in all datasets
agua = agua.rename(columns={'SIT_ATUAL': 'SIT', 'ANO_PLANTI': 'ANO'})

# Select columns UHE EUCLIDES CUNHA
euclides = gpd.read_file(list_shp[9]).loc[:, ['SIT_ATUAL',
                                              'ANO_PLANTI',
                                              'geometry']].to_crs(4326)

# Filter based on platation year and current situation
euclides = euclides[(euclides['ANO_PLANTI'].astype(float) > 0) &
                    ((euclides['SIT_ATUAL'] == 'EM RESTAURAÇÃO') |
                     (euclides['SIT_ATUAL'] == 'RESTAURADA'))] \
                     .reset_index(drop=True)

euclides['ANO_PLANTI'] = euclides['ANO_PLANTI'].astype(int)

# Change the name to be the same in all datasets
euclides = euclides.rename(columns={'SIT_ATUAL': 'SIT', 'ANO_PLANTI': 'ANO'})

# Concatenate all datasets
aes_final = pd.concat([bariri, limoeiro, caconde, promissao,
                       mogi, barra, ibitinga, nova, agua, euclides],
                      axis=0, ignore_index=True)

# Ser year as int type
aes_final['ANO'] = aes_final.ANO.astype(int)

# Select only columns of interest
aes_final = aes_final.loc[:, ['ANO', 'geometry']]

# filter polygons that are inside atlantic rainforest
aes_final = aes_final[aes_final.within(atlantic.geometry.iloc[0])]

# Rename year columns to be consistent with other datasets
aes_final = aes_final.rename(columns={'ANO': 'ano'})

# Save final dataset
aes_final.to_file('/maps/fnb25/data/polygons_filtered/aes.gpkg', fid=False)
