# Title: Understanding Drought Awareness from Web Data

We used computer vision (U-Net) model to leverage Standardized Precipitation Evapotranspiration Index (SPEI), Google Trends Search Interest (SI), and Twitter data to understand patterns with which people in Continental United States (CONUS) indicate awareness of and interest in droughts.

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

11_trend_analysis.py - This script performs trend analysis using the SPEI and SI datasets and generates figures. 

12_tweet_sentiment_analysis.py - This script performs sentiment analysis on Twitter data and creates a time series of the percentage of different sentiments.
