import os
import sys
from pathlib import Path


# Patch os.add_dll_directory
original_add_dll_directory = os.add_dll_directory

def safe_add_dll_directory(p):
    if os.path.isabs(p) and os.path.isdir(p):
        return original_add_dll_directory(p)
    return None

os.add_dll_directory = safe_add_dll_directory


# Define QGIS install location
QGIS_PATH = r"C:\Program Files\QGIS 3.34.12"

# Construct safe DLL search paths
dll_paths = [
    os.path.join(QGIS_PATH, 'bin'),
    os.path.join(QGIS_PATH, 'apps', 'qgis-ltr', 'bin'),
    os.path.join(QGIS_PATH, 'apps', 'Qt5', 'bin')
]

# Filter existing PATH for valid absolute directories only
original_path = os.environ.get('PATH', '')
valid_paths = [p for p in original_path.split(';') if os.path.isabs(p) and os.path.isdir(p)]
os.environ['PATH'] = ';'.join(dll_paths + valid_paths)

# Set GDAL data path to suppress some errors
os.environ['GDAL_DATA'] = os.path.join(QGIS_PATH, 'share', 'gdal')

# Add QGIS Python modules to sys.path
sys.path.append(os.path.join(QGIS_PATH, 'apps', 'qgis-ltr', 'python'))
sys.path.append(os.path.join(QGIS_PATH, 'apps', 'qgis-ltr', 'python', 'plugins'))
sys.path.append(os.path.join(QGIS_PATH, 'apps', 'Python312', 'Lib', 'site-packages'))

# Add UMEP plugin path
sys.path.append(r"C:\Users\Ardo\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins")

# Set QGIS environment variables
os.environ['QGIS_PREFIX_PATH'] = os.path.join(QGIS_PATH, 'apps', 'qgis-ltr')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(QGIS_PATH, 'apps', 'Qt5', 'plugins')


# Initialize QGIS
from qgis.core import QgsApplication
from PyQt5.QtCore import QDate, QTime

qgs = QgsApplication([], False)
qgs.initQgis()
print("QGIS initialized successfully with Python 3.12")

# Initialize Processing + UMEP
from processing.core.Processing import Processing
Processing.initialize()

from processing_umep.processing_umep_provider import ProcessingUMEPProvider
umep_provider = ProcessingUMEPProvider()
QgsApplication.processingRegistry().addProvider(umep_provider)

import processing

# Loop through folders and run tools
base_dir = Path(r'../dataset/patches')

try:
    for folder in base_dir.iterdir():
        if folder.is_dir():
            dsm_path = folder / 'dsm.tif'
            dem_path = folder / 'dem.tif'
            cdsm_path = folder / 'cdsm.tif'
            svf_output_path = folder / 'svf.tif'
            wall_height_path = folder / 'wall_height.tif'
            wall_aspect_path = folder / 'wall_aspect.tif'
            landcover_path = folder / 'combined_landuse.tif'
            shadow_path = folder / 'shadow.tif'
            #building_path = folder / 'buildings.tif'
            

            if not dsm_path.exists():
                print(f"Skipped {folder.name}: dsm.tif not found")
                continue

            print(f"\n Processing folder: {folder.name}")

            try:
                svf_output = processing.run("umep:Urban Geometry: Sky View Factor", {
                    'INPUT_DSM': str(dsm_path),
                    'INPUT_CDSM': str(cdsm_path),
                    'TRANS_VEG': 10,
                    'INPUT_TDSM': None,
                    'INPUT_THEIGHT': 25,
                    'ANISO': True,
                    'WALL_SCHEME': False,
                    'KMEANS': True,
                    'CLUSTERS': 5,
                    'INPUT_DEM': None,
                    'INPUT_SVFHEIGHT': 1,
                    'OUTPUT_DIR': str(folder),
                    'OUTPUT_FILE': str(svf_output_path)
                })
                print(f"SVF created: {svf_output_path.name}")

                wall_outputs = processing.run("umep:Urban Geometry: Wall Height and Aspect", {
                    'INPUT': str(dsm_path),
                    'INPUT_LIMIT': 3,
                    'OUTPUT_HEIGHT': str(wall_height_path),
                    'OUTPUT_ASPECT': str(wall_aspect_path)
                })
                print("Wall height and aspect done")

                shadow_output = processing.run("umep:Solar Radiation: Shadow Generator", {
                    'INPUT_DSM':str(dsm_path),
                    'INPUT_CDSM':str(cdsm_path),
                    'TRANS_VEG':10,
                    'INPUT_TDSM':None,
                    'INPUT_THEIGHT':25,
                    'INPUT_HEIGHT': str(wall_height_path),
                    'INPUT_ASPECT': str(wall_aspect_path),
                    'UTC':1,
                    'DST':False,
                    'DATEINI':QDate(2025, 8, 3),
                    'ITERTIME':60,
                    'ONE_SHADOW':False,
                    'TIMEINI':QTime(18, 8, 7),
                    'OUTPUT_DIR':'TEMPORARY_OUTPUT',
                    'OUTPUT_FILE':str(shadow_path)
                })
                print("Shadow done")

                solweig_output = processing.run("umep:Outdoor Thermal Comfort: SOLWEIG", {
                    'INPUT_DSM': str(dsm_path),
                    'INPUT_SVF': str(folder / 'svfs.zip'),
                    'INPUT_HEIGHT': str(wall_height_path),
                    'INPUT_ASPECT': str(wall_aspect_path),
                    'INPUT_CDSM': str(cdsm_path),
                    'TRANS_VEG': 10,
                    'LEAF_START': 97,
                    'LEAF_END': 300,
                    'CONIFER_TREES': False,
                    'INPUT_TDSM': None,
                    'INPUT_THEIGHT': 25,
                    'INPUT_LC': str(landcover_path),
                    'USE_LC_BUILD': True,
                    'INPUT_DEM': str(dem_path),
                    'SAVE_BUILD': True,
                    'INPUT_ANISO': '',
                    'INPUT_WALLSCHEME': '',
                    'WALLTEMP_NETCDF': False,
                    'WALL_TYPE': 0,
                    'ALBEDO_WALLS': 0.2,
                    'ALBEDO_GROUND': 0.15,
                    'EMIS_WALLS': 0.9,
                    'EMIS_GROUND': 0.95,
                    'ABS_S': 0.7,
                    'ABS_L': 0.95,
                    'POSTURE': 0,
                    'CYL': True,
                    'INPUTMET': r'C:\Users\Ardo\Desktop\thesis2\climate_BCN_1985-08-03_24h.txt',
                    'ONLYGLOBAL': False,
                    'UTC': 1,
                    'WOI_FILE': None,
                    'WOI_FIELD': '',
                    'POI_FILE': None,
                    'POI_FIELD': '',
                    'AGE': 35,
                    'ACTIVITY': 80,
                    'CLO': 0.9,
                    'WEIGHT': 75,
                    'HEIGHT': 180,
                    'SEX': 0,
                    'SENSOR_HEIGHT': 10,
                    'OUTPUT_TMRT': False,
                    'OUTPUT_KDOWN': False,
                    'OUTPUT_KUP': False,
                    'OUTPUT_LDOWN': False,
                    'OUTPUT_LUP': False,
                    'OUTPUT_SH': False,
                    'OUTPUT_TREEPLANTER': False,
                    'OUTPUT_DIR': str(folder)
                })
                print("MRT (SOLWEIG) completed")

                """tmrt_output = processing.run("umep:Outdoor Thermal Comfort: SOLWEIG Analyzer",{
                    'SOLWEIG_DIR': str(folder),
                    'BUILDINGS':'C:/Users/Ardo/Desktop/thesis2/mrt/buildings.tif',
                    'VARIA_IN':0,
                    'STAT_TYPE':1,
                    'THRES_TYPE':0,
                    'TMRT_THRES_NUM':55,
                    'STAT_OUT':'TEMPORARY_OUTPUT',
                    'TMRT_STAT_OUT':'TEMPORARY_OUTPUT'
                    })
                print("MeanRadiantTemperature completed")"""

            except Exception as e:
                print(f"Error processing {folder.name}: {e}")

except KeyboardInterrupt:
    print("Interrupted by user")

# Clean exit
qgs.exitQgis()
print("QGIS session closed.")