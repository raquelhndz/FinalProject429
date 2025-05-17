# Project Summary

The goal of my project was to pair students in 6th-12th grade with nonprofit organizations based on their location and interests, helping them find potential local partners to collaborate with on inventions for social good. I created an application that suggests five nearby organizations based on a user’s input sentence describing what they’re interested in.

[**Final Product**](https://community-matching-4030229048.us-central1.run.app/)  
*Please note it takes a little while to load because it accesses a large dataset (almost one million rows!).*

[**Final Presentation Slides:**](https://docs.google.com/presentation/d/1lH2ij4ymx_ZaTJVi78ol9RbpHtFo0rSmF8Q-owhMWJw/edit?usp=sharing)  
*These slides give an overview of my project process and the technologies I used.*

## Original Datasets I used:
- [IRS Publication 78 Data 2024](https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads#pub78): List of organizations eligible to receive tax-deductible charitable contributions.  
- [Urban Data Catalog CBDO Population Dataset.csv 2022](https://datacatalog.urban.org/dataset/community-based-development-organization-sector-and-financial-datasets): Community-Based Development Organizations Sector.

## My Datasets (Google Drive Links):
- [df_coordinates_cleaned.csv](https://drive.google.com/file/d/1UhCQyuGXV96iBc_4m1rgy1deUaGNKXm6/view?usp=sharing): Cities with latitude and longitude coordinates.  
- [df_api_final.csv](https://drive.google.com/file/d/1xv7R0Fm4Ss1ia8PnkPx9nOCWnu71mLbi/view?usp=sharing): Organizations with text descriptions from the Charity Navigator API.  
- [final_dataset.csv](https://drive.google.com/file/d/1OD9l3hHe_LHZ8b4lSXMNEJFA2ATBX5lR/view?usp=sharing): Organizations with text descriptions and BERT sentence embedding vectors (very large file).

---

# Overview of Files in the Repository

In the **Development Process** folder, I included the Google Colab notebooks I used while developing the final product:

- **CharityNavigatorAPIScript.ipynb**  
  Developed a script to call the Charity Navigator API, which I included in other notebooks when needed.

- **AI Studio Project Data Cleaning.ipynb**  
  Combined the initial datasets (CBDO and IRS Pub78) and performed initial data cleaning. I pulled information about each organization’s purpose from NTEE codes and the Charity Navigator API. Since the dataset was very large, I ran the API in chunks—only the first chunk is shown here.

- **DataPreparationAndModeling.ipynb**  
  Prepared data for modeling by combining API data chunks, dropping unnecessary columns, removing duplicates, and handling missing values.

- **BTT_modeling.ipynb**  
  Explored the modeling process using BERT Sentence Transformers.

- **GeoCode Example.ipynb**  
  Learned how to use the GeoPy library.

- **Location_work.ipynb**  
  Created a dataset of cities with their latitude and longitude coordinates for later use.

- **KMeansCluster_location.ipynb** (too big to upload to GitHub, link [here](https://colab.research.google.com/drive/12bjS2ENHvrfypCdVd7gqKxblHqI6b7nh?authuser=2#scrollTo=i1O4bM_1374H))  
  Experimented with location matching using K-means clustering but did not use this method due to efficiency concerns.

- **DashInterfaceWork.ipynb**  
  Created the UI for my application using Dash.

- **ModelFullDataSet.ipynb**  
  Saved sentence embedding vectors of each organization’s description in the dataframe and ran the model on the complete dataset.

In the **org_matching** folder, I included the final files for deployment. The python file can be run locally or deployed on Google Cloud Run:

- **dashinterfacework.py**  
  Final application code: a Dash interface that suggests organizations based on location and interest inputs.

- **requirements.txt**  
  Package requirements for dashinterfacework.py.

- **Dockerfile**  
  Used to create a Docker image for uploading to Google Cloud Run.
