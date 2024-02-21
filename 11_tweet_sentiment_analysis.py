#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl
import matplotlib.pyplot as plt
import pingouin as pg
import json
import ast
from shapely.geometry import Point
import geopandas as gpd
import contextily as ctx
from tqdm import tqdm
import swifter
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr
# from mpl_toolkits.basemap import Basemap


# In[ ]:


#Load dataframe 
with open('data/drought_twitter_corpus_df.pkl', 'rb') as f:
    corpus_df = pkl.load(f)


# In[ ]:


corpus_df.columns


# In[ ]:


# sentiment_list = []

# def percentage(part,whole):
#  return 100 * float(part)/float(whole)

# # noOfTweet = int(input ('Please enter how many tweets to analyze: '))

# positive = 0
# negative = 0
# neutral = 0
# polarity = 0
# tweet_list = []
# neutral_list = []
# negative_list = []
# positive_list = []

# for twt in corpus_df['tweet']:
# #     print(twt)
    
#     tweet_list.append(twt)
#     analysis = TextBlob(twt)
#     score = SentimentIntensityAnalyzer().polarity_scores(twt)
#     neg = score['neg']
#     neu = score['neu']
#     pos = score['pos']
#     comp = score['compound']
#     polarity += analysis.sentiment.polarity
 
#     if neg > pos:
#         sentiment_list.append('neg')
# #         negative_list.append(twt)
# #         negative += 1
#     elif pos > neg:
#         sentiment_list.append('pos')
# #         positive_list.append(twt)
# #         positive += 1
#     elif pos == neg:
#         sentiment_list.append('neu')
# #         neutral_list.append(twt)
# #         neutral += 1
# # positive = percentage(positive, noOfTweet)
# # negative = percentage(negative, noOfTweet)
# # neutral = percentage(neutral, noOfTweet)
# # polarity = percentage(polarity, noOfTweet)
# # positive = format(positive, '.1f')
# # negative = format(negative, '.1f')
# # neutral = format(neutral, '.1f')


# In[ ]:


# senti_arr = np.array(sentiment_list)
# np.save('data/senti_array', senti_arr, allow_pickle=True, fix_imports=True)


# In[ ]:


# #Number of Tweets (Total, Positive, Negative, Neutral)
# tweet_list = pd.DataFrame(tweet_list)
# neutral_list = pd.DataFrame(neutral_list)
# negative_list = pd.DataFrame(negative_list)
# positive_list = pd.DataFrame(positive_list)
# print('total number: ',len(tweet_list))
# print('positive number: ',len(positive_list))
# print('negative number: ', len(negative_list))
# print('neutral number: ',len(neutral_list))


# In[ ]:


# #Creating PieCart
# labels = ['Positive [‘+str(positive)+’%]' , 'Neutral [‘+str(neutral)+’%]','Negative [‘+str(negative)+’%]']
# sizes = [positive, neutral, negative]
# colors = ['yellowgreen', 'blue','red']
# patches, texts = plt.pie(sizes,colors=colors, startangle=90)
# plt.style.use('default')
# plt.legend(labels)
# plt.title('Sentiment Analysis Result for keyword= “+Drought+”')
# plt.axis('equal')
# plt.show()


# In[ ]:


senti_arr = np.load('data/senti_array.npy')


# In[ ]:


senti_list = senti_arr.tolist()


# In[ ]:


corpus_df['sentiment'] =  senti_list


# In[ ]:


np.unique(corpus_df['geo'])[-5:]


# In[ ]:


np.unique(corpus_df['geo'])[-1:][0]


# In[ ]:


# Adjusted function to check if geo value is in the expected GeoJSON-like format
def is_geojson(geo_str):
    try:
        # Safely evaluate the string as a Python literal (dictionary)
        geo_dict = ast.literal_eval(geo_str)
        # Check if 'coordinates' is a key in the nested dictionary
        if 'coordinates' in geo_dict.get('coordinates', {}):
            return True
    except:
        # If there's an error in parsing or converting, it's not in the expected format
        return False

# Apply the adjusted function to create a mask for rows with GeoJSON-like data
geojson_mask = corpus_df['geo'].apply(is_geojson)


# In[ ]:


# Count the number of True values in the geojson_mask
geojson_count = geojson_mask.sum()

geojson_count


# In[ ]:


# Filter the original DataFrame using the geojson_mask to create a new DataFrame with GeoJSON-like data
filtered_geojson_df = corpus_df[geojson_mask].copy()


# In[ ]:


# Function to extract latitude and longitude from the GeoJSON-like string
def extract_lat_lon(geo_str):
    try:
        # Convert the string representation of dictionary into an actual dictionary
        geo_dict = ast.literal_eval(geo_str)
        # Extract latitude and longitude
        coordinates = geo_dict['coordinates']['coordinates']
        # Note: GeoJSON format is [longitude, latitude]
        return pd.Series([coordinates[1], coordinates[0]])  # Return as latitude, longitude
    except Exception as e:
        # In case of any error, return NaN values
        return pd.Series([pd.NA, pd.NA])

# Apply the function to the 'geo' column and create new columns for latitude and longitude
filtered_geojson_df[['latitude', 'longitude']] = filtered_geojson_df['geo'].apply(extract_lat_lon)


# In[ ]:


np.unique(filtered_geojson_df['longitude'])


# In[ ]:


corpus_df['created_at'][:5]


# In[ ]:


# # To list all available providers:
# print(ctx.providers.keys())

# ig, ax = plt.subplots(figsize=(10, 10))

# # Plot points with different colors based on sentiment
# for sentiment, color in [('pos', 'green'), ('neg', 'red'), ('neu', 'gray')]:
#     gdf[gdf['sentiment'] == sentiment].plot(ax=ax, markersize=5, color=color, label=sentiment)

# # Add basemap using OpenStreetMap
# ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# # Set the bounds for CONUS (these are approximate and can be adjusted)
# # The bounds are in Web Mercator projection (EPSG:3857)
# ax.set_xlim(-13884029, -7453304)
# ax.set_ylim(2810491, 6338219)

# # Legend and settings
# ax.legend()
# ax.set_axis_off()

# plt.title("Tweet Sentiments Across the CONUS")
# plt.show()


# In[ ]:


# Define US geographic boundaries
min_lat, max_lat = 24.396308, 49.384358
min_lon, max_lon = -125.0, -66.93457

# Filter rows where latitude and longitude are within the US boundaries
us_tweets_df = filtered_geojson_df[
    (filtered_geojson_df['latitude'] >= min_lat) & (filtered_geojson_df['latitude'] <= max_lat) &
    (filtered_geojson_df['longitude'] >= min_lon) & (filtered_geojson_df['longitude'] <= max_lon)
]


# In[ ]:


us_tweets_df['created_at'] = pd.to_datetime(us_tweets_df['created_at'])
us_tweets_df['year'] = us_tweets_df['created_at'].dt.year
us_tweets_df = us_tweets_df.sort_values(by='created_at').reset_index(drop=True)


# In[ ]:


us_tweets_df 


# In[ ]:


# Ensure 'created_at' is a datetime column
corpus_df['created_at'] = pd.to_datetime(corpus_df['created_at'])

# Extract the year from 'created_at' and create a new column 'year'
corpus_df['year'] = corpus_df['created_at'].dt.year

# Group by 'year' and sample 1%
sampled_df = corpus_df.groupby('year').apply(lambda x: x.sample(frac=0.2)).reset_index(drop=True)


# In[ ]:


geonames_df = pd.read_csv('US/US.txt', sep='\t', dtype=str, names=[
    'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
    'feature class', 'feature code', 'country code', 'cc2', 'admin1 code',
    'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation',
    'dem', 'timezone', 'modification date'
], na_values=[''])


# In[ ]:


# Adjusted function to check for GeoNames 
def matches_geonames(text, geonames_df):
    for place in geonames_df['name'].unique():
        if place.lower() in text.lower():
            return True
    return False

#Apply function with Swifter for automatic parallelization
sampled_df['location_in_us'] = sampled_df['geo'].swifter.apply(lambda x: matches_geonames(x, geonames_df) if pd.notnull(x) else False)
sampled_df['tweet_in_us'] = sampled_df['tweet'].swifter.apply(lambda x: matches_geonames(x, geonames_df))
sampled_df['in_us'] = sampled_df['location_in_us'] | sampled_df['tweet_in_us']
sampled_df = sampled_df[sampled_df['in_us']]


# In[ ]:


# Concatenate us_tweets_df with sampled_df
combined_df = pd.concat([sampled_df, us_tweets_df], ignore_index=True)

# Remove duplicates based on the 'id' column, keeping the first occurrence
combined_df = combined_df.drop_duplicates(subset=['id'])

# Ensure 'created_at' is in datetime format
combined_df['created_at'] = pd.to_datetime(combined_df['created_at'])

# Sort by 'created_at' and reset index
combined_df = combined_df.sort_values(by='created_at').reset_index(drop=True)


# In[ ]:


combined_df


# In[ ]:


# Group by year and sentiment, count occurrences, and then unstack to create a wide format DataFrame
sentiment_counts_yearly = combined_df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)

# Calculate percentages of each sentiment per year
sentiment_percentages_yearly = sentiment_counts_yearly.div(sentiment_counts_yearly.sum(axis=1), axis=0) * 100

# Filter the DataFrame for years 2008 to 2020
filtered_sentiment_percentages_yearly = sentiment_percentages_yearly.loc[2008:2020]


# In[ ]:


filtered_sentiment_percentages_yearly


# In[ ]:


# Increase the base font size
plt.rcParams.update({'font.size': 16})

# Create a figure with two subplots (side by side)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Plot the time series in the first subplot
for sentiment in filtered_sentiment_percentages_yearly.columns:
    axes[0].plot(filtered_sentiment_percentages_yearly.index, filtered_sentiment_percentages_yearly[sentiment], linestyle='-', linewidth=5, label=sentiment)

axes[0].set_title('Yearly Sentiment Percentages over United States')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Percentage')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True)
axes[0].legend(title='Sentiment')

# Calculate overall sentiment percentages for the pie chart
overall_sentiments = us_tweets_df['sentiment'].value_counts(normalize=True) * 100

# Plot the pie chart in the second subplot
axes[1].pie(overall_sentiments, labels=overall_sentiments.index, autopct='%1.0f%%', startangle=140, colors=['orange', 'green', 'blue'])
axes[1].set_title('Overall Sentiment Percentages \nover United States')

# Adjust layout to make room for the rotated x-axis labels and ensure no overlap
plt.tight_layout()

# Save the figure
plt.savefig('panel_6.png')

# Show the figure
plt.show()


# In[ ]:


# Convert index to a column for regression analysis
df = filtered_sentiment_percentages_yearly.reset_index()

def analyze_trend_pearson(df, sentiment, inflection_year=2014):
    # Split the data
    pre_data = df[df['year'] < inflection_year]
    post_data = df[df['year'] >= inflection_year]
    
    # Calculate Pearson correlation coefficient for pre-2014
    pre_corr_coeff, pre_p_value = pearsonr(pre_data['year'], pre_data[sentiment])
    
    # Calculate Pearson correlation coefficient for post-2014
    post_corr_coeff, post_p_value = pearsonr(post_data['year'], post_data[sentiment])
    
    # Print results
    print(f"Pre-2014 {sentiment} Sentiment: Pearson r = {pre_corr_coeff:.4f}, p-value = {pre_p_value:.4g}")
    print(f"Post-2014 {sentiment} Sentiment: Pearson r = {post_corr_coeff:.4f}, p-value = {post_p_value:.4g}")

# Apply the function for each sentiment
sentiments = ['neg', 'neu', 'pos']
for sentiment in sentiments:
    analyze_trend_pearson(df, sentiment)


# In[ ]:


# #Analyse the variability of sentiments

# #takes all unique values in data - year as well as how often they occur and returns them as an array.
# uniqueyears, time_slices = np.unique(corpus_df['Month'], return_counts=True) 

# time_slices = time_slices.tolist()

# def cumSum(s):
#     sm=0
#     cum_list=[]
#     for i in s:
#         sm=sm+i
#         cum_list.append(sm)
#     return cum_list

# time_slices_cum = cumSum(time_slices)
# time_slices_cum


# In[ ]:


# #Add 0 as the first element of the cumulative document array
# a=0
# time_slices_cum.insert(0,a)
# time_slices_cum


# In[ ]:


# def percentage(part,whole):
#     return 100 * float(part)/float(whole)


# pos_list = []
# neg_list = []
# neu_list = []




# for t in range(len(time_slices)):
    
#     positive = 0
#     negative = 0
#     neutral = 0
    
#     for i in range(time_slices_cum[t], time_slices_cum[t+1]):
#         if str(corpus_df['sentiment'][i]) == 'pos':
#             positive += 1
#         elif str(corpus_df['sentiment'][i]) == 'neg':
#             negative += 1
#         elif str(corpus_df['sentiment'][i]) == 'neu':
#             neutral += 1

#     pos = percentage(positive, time_slices[t])
#     neg = percentage(negative, time_slices[t])
#     neu = percentage(neutral, time_slices[t])
    
#     pos_list.append(pos)
#     neg_list.append(neg)
#     neu_list.append(neu)


# In[ ]:


# #Pull months
# months = np.unique(corpus_df['Month'])


# In[ ]:


# #Plot sentiment variability

# n = 10  # Show every nth label
# plt.figure(figsize=(50,15))
# plt.plot(months, pos_list, color = 'green', label = 'Positive', linewidth = 5)
# plt.plot(months, neg_list, color = 'red', label = 'Negative', linewidth = 5)
# plt.plot(months, neu_list, color = 'orange', label = 'Neutral', linewidth = 5)

# # Reduce the density of xticks
# ticks = plt.gca().get_xticks()
# ticklabels = [months[int(i)] if i < len(months) and i%10 == 0 else '' for i in ticks]

# plt.gca().set_xticklabels(ticklabels, fontsize=40, rotation=90)

# plt.yticks(size = 40)
# plt.xlim(0,155)
# plt.grid()
# plt.xlabel("Months", fontsize = 40)
# plt.ylabel("Sentiment Percentages", fontsize = 40)
# plt.legend(prop={'size': 40}, fancybox = True, loc = 'upper left')
# plt.title("Variability of Public Sentiment of Drought from Twitter Data", size = 45)
# plt.savefig('sentiment_variability')
# plt.show()


# In[ ]:


# #Load Worldwide Google Search Interest for Drought 
# google_drought = pd.read_csv('data/google_drought_worldwide.csv')
# google_climatechange = pd.read_csv('data/climatechange_google.csv')
# google_waterscarcity = pd.read_csv('data/waterscarcity_google.csv')
# google_drought = google_drought['Category: All categories'][1:]
# google_climatechange = google_climatechange['Category: All categories'][1:]
# google_waterscarcity = google_waterscarcity['Category: All categories'][1:]


# In[ ]:


# google_drought_list = []
# google_climatechange_list = []
# google_waterscarcity_list = []

# for d in range(len(google_drought)):
    
#     google_drought_list.append(int(google_drought[d]))
#     google_climatechange_list.append(int(google_climatechange[d]))
#     google_waterscarcity_list.append(int(google_waterscarcity[d]))


# In[ ]:


# fig = plt.figure()
# fig, ax1 = plt.subplots(figsize=(50, 15))
# ax2 = ax1.twinx()

# ax1.tick_params(axis='both', which='major', labelsize=25, rotation = 90)
# ax2.tick_params(axis='both', which='major', labelsize=25)

# ax1.plot(months, pos_list, color = 'green', label = 'Positive', linewidth = 5)
# ax1.plot(months, neg_list, color = 'red', label = 'Negative', linewidth = 5)
# ax1.plot(months, neu_list, color = 'orange', label = 'Neutral', linewidth = 5)
# ax2.plot(months, google_drought_list, color = 'skyblue', linestyle='dashed', linewidth = 7, label = 'Google Drought SI')
# ax2.plot(months, google_climatechange_list, color = 'gray', linestyle='dashed', linewidth = 7, label = 'Google Climate Change SI')
# ax2.plot(months, google_waterscarcity_list, color = 'blue', linestyle='dashed', linewidth = 7, label = 'Google Water Scarcity SI')

# plt.xlim(0,168)
# ax1.set_ylabel('Sentiment Percentage', size = 40)
# ax2.set_ylabel('Google SI', size = 40)
# ax1.set_xlabel('Months', size = 40)
# ax1.grid()
# fig.legend(loc = 'upper right', fancybox = True, shadow = True, prop={'size': 20})
# plt.title('Variability of Global Twitter Public Sentiment and Google Public Search Interest on Drought', fontsize=50)


# In[ ]:


# #Test for statistical relationship between Twitter Sentiment and Google Search Interest

# print('Positive Sentiment & Google Drought SI')
# pg.corr(np.array(pos_list), np.array(google_drought_list))


# In[ ]:


# print('Negative Sentiment & Google Drought SI')
# pg.corr(np.array(neg_list), np.array(google_drought_list))


# In[ ]:


# print('Neutral Sentiment & Google Drought SI')
# pg.corr(np.array(neu_list), np.array(google_drought_list))


# In[ ]:


#Statistically significant weak positive correlation between google drought SI and polar sentiments
#Statistically significant moderate negative correlation between google SI and neutrality
#It appears polarization is statistically correlated to Google SI


# In[ ]:


# print('Positive Sentiment & Google Climate Change SI')
# pg.corr(np.array(pos_list), np.array(google_climatechange_list))


# In[ ]:


# print('Negative Sentiment & Google Climate Change SI')
# pg.corr(np.array(neg_list), np.array(google_climatechange_list))


# In[ ]:


# print('Neutral Sentiment & Google Climate Change SI')
# pg.corr(np.array(neu_list), np.array(google_climatechange_list))


# In[ ]:


# print('Positive Sentiment & Google Water Scarcity SI')
# pg.corr(np.array(pos_list), np.array(google_waterscarcity_list))


# In[ ]:


# print('Negative Sentiment & Google Water Scarcity SI')
# pg.corr(np.array(neg_list), np.array(google_waterscarcity_list))


# In[ ]:


# print('Neutral Sentiment & Google Water Scarcity SI')
# pg.corr(np.array(neu_list), np.array(google_waterscarcity_list))

