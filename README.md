# CS210 - Data Science Project
https://antistrange05-cs210-app-livadq.streamlit.app/

This project investigates the relationship between geographic isolation and lexical divergence across the world's language families. It uses a pairwise methodology to treat 15,183 language pairs as independent observations. It includes data preprocessing, spatial feature engineering, OLS regression and Mantel tests, and a Streamlit web application for interactive visualization.

## Project Goals

• Clean and preprocess ASJP vocabulary data and Glottolog geographic coordinates
• Generate valid related language pairs within families to prevent masking variance
• Build a multiple OLS regression model to predict linguistic divergence
• Analyze the "terrain paradox" using terrain ruggedness indices
• Create an interactive Streamlit application to visualize

Features

Data Cleaning & Preprocessing

• Handle missing coordinate data and pivot ASJP phonetic forms into a matrix
• Drop irrelevant language families and filter for families with 2 to 50 languages

Geospatial Feature Engineering

• Calculate great-circle distance (km) between homelands using the Haversine formula
• Sample Nunn & Puga ruggedness index via GeoPandas along a 5-point LineString

Statistical Analysis

• Run Spearman correlations to test distance and ruggedness against divergence
• Perform a permutation-based Mantel test (n=999) to account for non-independence
• Train a multiple OLS regression model

Streamlit Web App

• Generate real-time lexical divergence predictions between any two map coordinates
• Dynamically plot geographic corridors and nearest real-world language pairs
• Filter and explore 15,183 language pairs with live-updating regression plots

Jupyter Notebooks

• Compute the full data pipeline from raw datasets to the final pairs.csv
• Visualize world maps of within-family divergence and regional boxplots

Project Structure




README.md

requirements.txt


LICENSE

Datasets/

forms.csv
languages.csv
languoid.csv
rugged_data.csv
pairs.csv



☐ app.py

MAIN Notebooks/

☐ linguistic_geography.ipynb

How to run - Navigate to linguistic_geography.ipynb and hit the Colab button on the top left hand corner, everything is laid out there!
Also, my streamlit app is linked at the top of this page to run my app.py



