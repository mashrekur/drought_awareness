#Drought Awareness over Continental United States

Understanding the relationship between droughts and drought awareness is vital towards decision making and policy for water management and conservation strategies, and socioeconomic outcomes. We used computer vision (UNet models) to analyze nonlinear, lagged correlations between Standardized Precipitation Evapotranspiration Index (SPEI) and Google Trends Search Interest, and we used Twitter data to asses awareness and sentiments abut droughts within the Continental United States (CONUS). The most important drivers of this relationship are the variability and ranges of drought trends and severity, as well as climatic extremes.  This relationship was the strongest for Western states, followed by Northeastern, Southeastern, and Central regions. Search interest tends to lag droughts by a period of ~1-3 months. We also found evidence that reductionist linear approaches, such as a Principal Component Analysis, might not be as effective as UNet models in capturing the nuanced relationship between droughts and drought awareness at various dimensions and scales. We subsequently applied sentiment analysis on a set of 2.5 million georeferenced tweets related to droughts and found that people's sentiments towards drought have become increasingly positive with decreasing neutral sentiments since 2014 within the United States.
This repository contains the all the codes used in the study. The Google Trends, SPEI, and Twitter dataset used in this study have not been uploaded due to data and privacy restrictions.

[![DOI](https://zenodo.org/badge/674305649.svg)](https://zenodo.org/doi/10.5281/zenodo.8212807)

## Usage

1_get_trends_data.py - This script uses the Google Trends API to gather data about the search interest for the term "droughts" over a specified date range (from 2004 to 2022). The data is retrieved on a per-region basis, both for US and global regions. Then, using geopandas and matplotlib, the script generates choropleth maps that visually depict the search interest data for the term "droughts". 

 
2_get_twitter_data.py - Interacts with the Twitter API to fetch tweets about climate change from 2008 to 2022, using bearer tokens for authorization and handling pagination. Specific tweet data including the author ID, creation time, geo-tagging, tweet ID, language, count metrics (like, quote, reply, retweet), source and the tweet text itself are retrieved. Users must apply to Twitter to gain an authorization token.


3_make_spei_maps.py - This script processes the NetCDF file containing Standardized Precipitation & Evapotranspiration Index (SPEI) data, converts it to a pandas dataframe, cleans it, and visualizes the data. It then creates lagged SPEI maps for CONUS.

4_trends_api_map.py - This script creates search interest maps for CONUS using Google Trends search interest data.

5_visualize_UNET_architecture.py - Visualize the U-Net architecture before training.

6_UNET_CNN2D_CONUS.py - This script loads maps of Google search trends and SPEI and uses a U-Net model to find relationships between them. The script defines a function to load images from a directory, splits them into training and testing sets, defines the U-Net architecture, and trains the U-Net model. After training, it calculates the average R-squared score on the test set. It repeats this for different time lags, testing the model's performance when the trends data is shifted a certain number of months ahead of the SPEI data. Finally, it plots the R-squared value distributions for different lags, demonstrating how the predictive power of the model changes with different lags.

7_best_lag_heatmap.py - This script generates a heatmap visualizing the best-performing model at each pixel location and then by state.


8_UNET_CNN2D_CONUS_timeseries.py - This script generates a heatmap visualizing the spatial variability of the relationship between meteorological droughts and corresponding search interest.

9_UNET_CNN2D_CONUS_timeseries_statewise.py - This script generates a barplot visualizing the statewise variability of the relationship between meteorological droughts and corresponding search interest.

10_drought_hotspots.py - This script generates maps of SPEI over set time periods for CONUS.

11_tweet_sentiment_analysis.py - This script performs the geoencoding of tweets and the time series analysis of sentiments. 

12_pca.py - This script contains the codes for PCA on our data and also generates figures needed for 13_all_panels.py.

13_all_panels.py - This script has the codes for post-hoc analysis of the drought and drought awareness datasets and models. It also generates figure panels 4 to 8.

14_additional_analysis.py - This script performs additional analysis supplemental to 12_all_panels.py

15_rfr.py - This script performs the feature importance analysis 

16_pca.py - This script performs the PCA 


