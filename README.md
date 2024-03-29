<img src="https://github.com/fnincao/reforestation-CNN/assets/65984824/274959e3-9a09-4cca-a7c4-f298acd270e0" alt="University of Cambridge Logo" align="left" width="63">

<img src="https://github.com/fnincao/reforestation-CNN/assets/65984824/adf42702-0716-469d-85f1-453f653b25b9" alt="AI4ER Logo" align="right" width="80">

<br><br><br><br>

<h1 align="center">Leveraging U-NET to map Brazilian Atlantic Forest’s Active Restoration using Remote Sensing data</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8111751.svg)](https://doi.org/10.5281/zenodo.8111751)


University of Cambridge A14ER CDT MRes project by Felipe Begliomini

## Project Description

This project aims to facilitate the monitoring of active restoration efforts in the Atlantic Forest, a globally significant biodiversity hotspot facing severe degradation and deforestation. Active restoration, a widely employed strategy in the Atlantic Forest, involves human interventions to accelerate ecosystem recovery. By utilizing machine learning techniques on high-resolution remote sensing imagery, we have developed a method to automatically detect and classify active restoration areas. Through the integration of optical and RADAR data, our models show promising results for identifying these sites. This approach has the potential to contribute valuable insights for effective environmental restoration policies and overcome the current lack of a centralized database for restoration efforts in the Atlantic Forest.

## Data

• Reference Active Restoration polygons: The dataset used in this project to calibrate and teste the models comprises 9,556 polygons (approximately 213 km2) representing active restoration sites within the Atlantic Forest. The dataset was sourced from a consortium of Brazilian partners, including academia, private companies, state entities, and NGOs. However, due to access restrictions and sensitive information, **the specific location details and direct database access cannot be shared**.

• Planet Scope: Largest freely available high-resolution optical dataset for tropical regions (5m resolution) with four spectral bands (Blue, Green, Red, Near-Infrared). Covers 2015-2023, offering a decade-long record of landscape changes.

• Sentinel-1: C-Band SAR satellite (10m resolution) from the European Space Agency's Copernicus program. Provides Single co-polarization VV and Dual-band cross-polarization VH imagery with a 12-day temporal resolution.

• ALOS/PALSAR-2: L-Band SAR satellite capturing global imagery (25m resolution) since 2014. Yearly composites available (2014-2020) in Single co-polarization HH and Dual-band cross-polarization HV.

## Models

We propose three configurations based on the original U-NET architecture: U-NET RGBN, U-NET NDVI, and U-NET Fusion. Each one differs in the Satellite Remote Sensing (SRS) layers used as input data. In U-NET RGBN, the 4-band Planet image (Red, Green, Blue, Near-Infrared) 2020 yearly composite is utilized as input, along with the reference masks of active restoration sites. U-NET NDVI employs the Planet NDVI temporal 3-bands to generate the corresponding reference. The U-NET Fusion configuration introduces a novel modification to the U-NET structure, allowing for the inclusion of different SRS layers with varying spatial resolutions. In the top level, the same Planet NDVI image is inputted, and after reducing its spatial resolution by half using the Max Pooling operation, the Sentinel-1 image is concatenated at the second level. The same procedure is repeated with the ALOS/PALSAR-2 image at the third level.


![UNET_page-0001](https://github.com/fnincao/reforestation-CNN/assets/65984824/679184cc-19a8-4094-b4a4-c2a3e4b5fdb0)

## Results

We conducted two separete experiments, calibrating the models using two datasets: Region of Interest 1 (ROI 1), which comprised the all available polygons, and ROI 2, which focused on a subset of the dataset with higher-quality information polygons. The accuracy metrics for each ROI are provided below. To read more insights about the results, please refer to the accompanying report available at the GitHub repository.


### Accuracy Metrics - Region of Interest 1 (ROI 1)

The accuracy metrics for the experiment in ROI 1 are presented in the table below:

<div align="center">

| Model        | Overall Accuracy | Dice Score | Precision | Recall |
|--------------|-----------------|------------|-----------|--------|
| U-NET RGBN   | 0.97            | 0.23       | 0.48      | 0.25   |
| U-NET NDVI   | 0.97            | 0.16       | 0.52      | 0.16   |
| U-NET Fusion | 0.97            | 0.26       | 0.51      | 0.31   |

</div>

Also, some successful segmentation results from the experiment conducted in ROI 1 are presented below:

![restults_roi_1_A_page-0001](https://github.com/fnincao/reforestation-CNN/assets/65984824/10e4a20b-4404-4840-8be0-22266991bb6b)

### Accuracy Metrics - Region of Interest 2 (ROI 2)

The accuracy metrics for the experiment in ROI 2 are presented in the table below:

<div align="center">

| Model        | Overall Accuracy | Dice Score | Precision | Recall |
|--------------|-----------------|------------|-----------|--------|
| U-NET RGBN   | 0.96            | 0.40       | 0.52      | 0.39   |
| U-NET NDVI   | 0.95            | 0.37       | 0.47      | 0.38   |
| U-NET Fusion | 0.96            | 0.36       | 0.54      | 0.35   |

</div>

Also, some successful segmentation results from the experiment conducted in ROI 2 are presented below:

![restults_roi_2_A_page-0001](https://github.com/fnincao/reforestation-CNN/assets/65984824/f253dc5f-1e9d-42f8-99e6-6408cbb5ac64)


## Prerequisites

Before using this project, please ensure that you have completed the following prerequisites:

1. Create a Google Earth Engine Account: To access and utilize the satellite imagery used in this project, you will need to create an account on Google Earth Engine. Visit the [Google Earth Engine website](https://earthengine.google.com/) to sign up for an account if you don't already have one.

2. Register for Norway's International Climate and Forests Initiative Satellite Data Program: To access the Planet Scope dataset, you will need to register for an free account with Norway's International Climate and Forests Initiative Satellite Data Program. Please visit [NICFI website](https://www.planet.com/nicfi/) to register for an account and gain access to the dataset.

3. Link your Google Earth Engine account: Once you have registered for both Google Earth Engine and the Norway's International Climate and Forests Initiative Satellite Data Program, you will need to link your Google Earth Engine account with the program. [Follow the instructions](https://developers.planet.com/docs/integrations/gee/nicfi/)
 provided by the program to establish the connection and enable access to the Planet Scope dataset within Google Earth Engine.

## Usage

The models can be utilized for segmenting new images by following the notebook available in the `/notebooks` folder. This notebook has been specifically developed for use in Google Colaboratory, which offers free access to GPU resources and provides a readily available environment for Deep Learning frameworks.

## Acknowledgments


The author extends sincere gratitude to all the advisors and collaborators for their invaluable project guidance and technical support throughout the study:

•	Dr. David Commes, Dr. Charlotte Wheeler, and Dr. Srinivasan Keshav from the University of Cambridge

•	Dr. Pedro Henrique Santin Brancalion from the School of Agriculture at USP/ESALQ.

•	Dr. Paulo Guilherme Molin from the Federal University of São Carlos

•	Dr. David Moffat, Dr. James Harding, and Dr. Daniel Clewley from the Plymouth Marine Laboratory 

The author would also like to thank the NERC Earth Observation Data Acquisition and Analysis Service (NEODAAS) for access to compute resources for this study. Finally, the author gratefully acknowledges the support and funding provided by the Cambridge Center for Carbon Credits, which greatly contributed to the completion of this study. 



