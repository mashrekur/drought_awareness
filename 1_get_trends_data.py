#!/usr/bin/env python
# coding: utf-8

# In[230]:


import pprint
from apiclient.discovery import build
from pytrends.request import TrendReq
import pandas as pd
# from google.oauth2 import service_account
import geopandas as gpd
import matplotlib.pyplot as plt
# from geopandas.datasets import countries
from matplotlib.patches import Patch
import numpy as np
from shapely.geometry import Point
import os


# In[199]:


SERVER = 'https://trends.googleapis.com'

API_VERSION = 'v1beta'
DISCOVERY_URL_SUFFIX = '/$discovery/rest?version=' + API_VERSION
DISCOVERY_URL = SERVER + DISCOVERY_URL_SUFFIX


# In[200]:


# Build the service object with the appropriate credentials
service = build('trends', 'v1beta',
                developerKey='your_developer_key',
                discoveryServiceUrl=DISCOVERY_URL)


# In[201]:


# List all the attributes of the resource object
print(dir(service))


# In[202]:


print(dir(service.regions()))


# In[226]:


# Define the start and end dates
start_date = '2004-01'
end_date = '2022-12'


# In[234]:


# Define the date range for the loop
date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

# Define an empty list to store the data
data = []


# In[207]:


# Loop through the date range and get the search interest data for each month
for i in range(len(date_range) - 1):
    # Define the start and end dates for this iteration
    start = date_range[i].strftime('%Y-%m')
    end = date_range[i + 1].strftime('%Y-%m')
    
    # Get the search interest data for the term "droughts"
    regions = service.regions().list(term='/m/099lp', 
                                     restrictions_startDate=start,
                                     restrictions_endDate=end,
                                    restrictions_geo='US')
    response = regions.execute()
    print(response)
    
    # Extract the data for each region
    for region in response['regions']:
        region_code = region['regionCode']
        region_name = region['regionName']
        value = region['value']
        
        # Append the data to the list
        data.append([start, region_name, region_code, value])

# Convert the list to a pandas dataframe
us_df = pd.DataFrame(data, columns=['Month', 'Region', 'Region ID', 'Search Interest'])


# In[208]:


# Save the dataframe to a CSV file
us_df.to_csv('us_search_interest.csv', index=False)


# In[209]:


# Create a new dataframe with non-zero search interest values
usa_df = us_df[us_df['Search Interest'] != 0]

usa_df['Region'] = usa_df['Region'].replace('United States', 'United States of America')

# Print the new dataframe
print(usa_df.head())


# In[221]:


# Load the US states shapefile
us_map = gpd.read_file('cb_2018_us_state_500k.shp')

# Filter and rename columns in search interest data
june_df = usa_df[(usa_df['Month'] == '2020-06') & (usa_df['Search Interest'] != 0)]

# Extract state abbreviations from the 'Region ID' column
june_df['STUSPS'] = june_df['Region ID'].str.split('-').str[1]

# Merge the two data frames using the STUSPS column
merged_df = us_map.merge(june_df, on='STUSPS', how='left')

# Fill missing values with NaN and set geometry column to US geometry
merged_df = merged_df.fillna({'Search Interest': 0})
merged_df = merged_df.set_geometry(us_map.geometry)

# Create the choropleth map
ax = merged_df.plot(column='Search Interest', cmap='gray_r', figsize=(15, 10), edgecolor='black', legend=False)

# Set the plot title
ax.set_title('US Search Interest on Drought Topic in June 2020')

# Adjust plot limits to focus on the continental US
ax.set_xlim(-130, -65)
ax.set_ylim(24, 50)


# In[232]:


# # Define the date range
# start_date = pd.to_datetime('2004-01-01')
# end_date = pd.to_datetime('2020-12-01')
# date_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m')
# date_range


# In[223]:


# Loop through the date range and create choropleth maps
for date in date_range:
    # Load the US states shapefile
    us_map = gpd.read_file('cb_2018_us_state_500k.shp')

    # Filter and rename columns in search interest data
    june_df = usa_df[(usa_df['Month'] == date) & (usa_df['Search Interest'] != 0)]

    # Extract state abbreviations from the 'Region ID' column
    june_df['STUSPS'] = june_df['Region ID'].str.split('-').str[1]

    # Merge the two data frames using the STUSPS column
    merged_df = us_map.merge(june_df, on='STUSPS', how='left')

    # Fill missing values with NaN and set geometry column to US geometry
    merged_df = merged_df.fillna({'Search Interest': 0})
    merged_df = merged_df.set_geometry(us_map.geometry)

    # Create the choropleth map
    ax = merged_df.plot(column='Search Interest', cmap='gray_r', figsize=(15, 10), edgecolor='black', legend=False)

    # Set the plot title
#     ax.set_title('US Search Interest on Drought Topic in June 2020')

    # Adjust plot limits to focus on the continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)

    # Save the map as a file
    ax.get_figure().savefig(f'usa_maps_google/{date}_map.png')
    
    # Close the figure
    ax.get_figure().clf()


# In[235]:


# Loop through the date range and get the search interest data for each month
for i in range(len(date_range) - 1):
    # Define the start and end dates for this iteration
    start = date_range[i].strftime('%Y-%m')
    end = date_range[i + 1].strftime('%Y-%m')
    
    # Get the search interest data for the term "droughts"
    regions = service.regions().list(term='/m/099lp', 
                                     restrictions_startDate=start,
                                     restrictions_endDate=end)
    response = regions.execute()
    print(response)
    
    # Extract the data for each region
    for region in response['regions']:
        region_code = region['regionCode']
        region_name = region['regionName']
        value = region['value']
        
        # Append the data to the list
        data.append([start, region_name, region_code, value])

# Convert the list to a pandas dataframe
df = pd.DataFrame(data, columns=['Month', 'Region', 'Region ID', 'Search Interest'])


# In[236]:


# Save the dataframe to a CSV file
df.to_csv('global_search_interest.csv', index=False)


# In[237]:


df


# In[238]:


# Group by 'Region' and find the maximum 'Search Interest' value for each country
max_interest_df = df.groupby('Region')['Search Interest'].max().reset_index()

# Display the resulting DataFrame
max_interest_df


# In[240]:


np.unique(max_interest_df['Region'])


# In[243]:


np.where(max_interest_df['Region'] == 'Canada')


# In[245]:


max_interest_df[37:39]


# In[239]:


# Load the world map data using geopandas
world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

max_interest_df = max_interest_df.set_geometry(world_map.geometry)

# Create the choropleth map
ax = max_interest_df.plot(column='Search Interest', cmap='gray_r', figsize=(15, 10), edgecolor='black', legend=False)
ax.grid()


# In[154]:


# Create a new dataframe with non-zero search interest values
new_df = df[df['Search Interest'] != 0]

new_df['Region'] = new_df['Region'].replace('United States', 'United States of America')

# Print the new dataframe
print(new_df.head())


# In[155]:


len(new_df)


# In[196]:


# Load the world map data using geopandas
world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# filter and rename columns in search interest data
june_df = new_df[(new_df['Month'] == '2020-06') & (new_df['Search Interest'] != 0)]
june_df = june_df.rename(columns={'Region': 'name'})

# merge the two data frames using the name column
merged_df = world_map.merge(june_df, on='name', how='left')

# Fill missing values with NaN and set geometry column to world geometry
merged_df = merged_df.fillna({'Search Interest': 0})
merged_df = merged_df.set_geometry(world_map.geometry)

# Create the choropleth map
ax = merged_df.plot(column='Search Interest', cmap='gray_r', figsize=(15, 10), edgecolor='black', legend=True)

# Set the plot title
ax.set_title('Global Search Interest in June 2020')


# In[193]:


# Load the world map data using geopandas
world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# filter and rename columns in search interest data
june_df = new_df[(new_df['Month'] == '2022-06') & (new_df['Search Interest'] != 0)]
june_df = june_df.rename(columns={'Region': 'name'})

# Fill missing values with NaN and set geometry column to world geometry
merged_df = merged_df.fillna({'Search Interest': 0})
merged_df = merged_df.set_geometry(world_map.geometry)

# Create the choropleth map
ax = merged_df.plot(column='Search Interest', cmap='gray_r', figsize=(15, 10), edgecolor='black', legend=False)
ax.grid()

# Set the plot title
# ax.set_title('Global Search Interest in June 2022')


# In[194]:


# Load the world map data using geopandas
world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# filter and rename columns in search interest data
june_df = new_df[(new_df['Month'] == '2005-12') & (new_df['Search Interest'] != 0)]
june_df = june_df.rename(columns={'Region': 'name'})

# Fill missing values with NaN and set geometry column to world geometry
merged_df = merged_df.fillna({'Search Interest': 0})
merged_df = merged_df.set_geometry(world_map.geometry)

# Create the choropleth map
ax = merged_df.plot(column='Search Interest', cmap='gray_r', figsize=(15, 10), edgecolor='black', legend=True)
ax.grid()
# Set the plot title
ax.set_title('Global Search Interest in Jan 2005')


# In[158]:


np.unique(june_df['name'])


# In[159]:


np.unique(world_map['name'])


# In[195]:


# Loop through the date range and create choropleth maps
for date in date_range:
    
    # Load the world map data using geopandas
    world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    # Filter and rename columns in search interest data

    t_df = new_df[(new_df['Month'] == date) & (new_df['Search Interest'] != 0)]
    t_df = t_df.rename(columns={'Region': 'name'})

    # Merge the two data frames using the name column
    merged_df = world_map.merge(t_df, on='name', how='left')
    
    # Fill missing values with NaN and set geometry column to world geometry
    merged_df = merged_df.fillna({'Search Interest': 0})
    merged_df = merged_df.set_geometry(world_map.geometry)

    # Create the choropleth map
    ax = merged_df.plot(column='Search Interest', cmap='gray_r', figsize=(15, 10), edgecolor='black', legend=False)
    ax.grid()

    # Set the plot title
#     ax.set_title(f'Global Search Interest in {date}')
    
    # Save the map as a file
    ax.get_figure().savefig(f'globe_maps_google/{date}_map.png')
    
    # Close the figure
    ax.get_figure().clf()

