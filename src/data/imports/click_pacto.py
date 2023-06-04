'''File to import/filter the dataset provided by Click Pacto'''
import glob
import geopandas as gpd
import fiona

PATH_ATLANTIC = '/maps/fnb25/data/suport_files/Atlantica'
PATH_CLICK = '/maps/fnb25/data/polygons_original/SOS_new/CLICK PACTO/'

# Load atlantic rainforest boundaries
atlantic = gpd.read_file(PATH_ATLANTIC).to_crs(4326)

# enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['kml'] = 'rw'

# enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw'

years = [2015, 2016, 2017, 2018]
geometry = []
date = []

# Load the data inside the folder with years names
for year in years:
    files = glob.glob(PATH_CLICK + str(year) + '/*.kml')
    for file in files:
        temp_file = gpd.read_file(file, driver='KML')
        exp_temfile = temp_file.explode(index_parts=False)
        for value in exp_temfile.is_valid:
            if not value:
                exp_temfile['geometry'] = exp_temfile.geometry.buffer(0)
        for item in exp_temfile.geometry.tolist():
            geometry.append(item)
            date.append(year)

files_revised_2016 = glob.glob(PATH_CLICK + 'revisados/*.kml')

# open the folder 2016/revisados - Delete this if not useful
for file in files_revised_2016:
    temp_file = gpd.read_file(file, driver='KML')
    exp_temfile = temp_file.explode(index_parts=False)
    for value in exp_temfile.is_valid:
        if not value:
            exp_temfile['geometry'] = exp_temfile.geometry.buffer(0)
    for item in exp_temfile.geometry.tolist():
        geometry.append(item)
        date.append(2016)

files_revise_2017 = glob.glob('/maps/fnb25/data/polygons_original/SOS_new/CLICK PACTO/2017/Revisar/*.kml') # noqa E501

# open the folder 2017/revisar - Delete this if not useful
for file in files_revise_2017:
    temp_file = gpd.read_file(file, driver='KML')
    exp_temfile = temp_file.explode(index_parts=False)
    for value in exp_temfile.is_valid:
        if not value:
            exp_temfile['geometry'] = exp_temfile.geometry.buffer(0)
    for item in exp_temfile.geometry.tolist():
        geometry.append(item)
        date.append(2017)

# Load data into a pandas geodataframe
click_pacto = gpd.GeoDataFrame({'ano': date, 'geometry': geometry})

# Set the CRS
click_pacto.crs = 'epsg:4326'

# Clean point geometries from the dataset
click_pacto = click_pacto[click_pacto.geom_type != 'Point']

# Filter polygons outside atlantic rainforest
click_pacto = click_pacto[click_pacto.within(atlantic.geometry.iloc[0])] \
                          .reset_index(drop=True)

# Save final dataset
click_pacto.to_file('/maps/fnb25/data/polygons_filtered/click_pacto.gpkg',
                    fid=False)
