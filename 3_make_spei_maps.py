#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.colors import Normalize
from matplotlib.cm import RdYlBu
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib
import time
from geopy.geocoders import Nominatim
import geocoder
import re
import os
from shapely.geometry import Point
from shapely.geometry import Polygon
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp2d
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle


# Open the NETCDF file using xarray
ds = xr.open_dataset('data/spei01.nc')

# Convert the xarray dataset to a pandas dataframe
df = ds.to_dataframe()

df = df.unstack()

# Print the dataframe to check the conversion
# print(df)


# In[3]:


spei_df = df.dropna()
spei_df = spei_df['spei']


# In[4]:


spei_df.columns


# In[5]:


# # Convert DataFrame to xlsx
# spei_df.to_excel('data/spei.xlsx', index=False)


# In[6]:


# fill the missing values lon in each column with the forward fill method.

spei_df = spei_df.reset_index()

spei_df = spei_df.apply(lambda x: x.ffill())

spei_df


# In[7]:


spei_df.columns


# In[8]:


# Plot global historic drought data

df2 = spei_df.mean(axis=0)
plt.figure(figsize=(25,6))
df2[2:].plot()
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlim(0,1440)
plt.grid()
plt.xlabel("Time", fontsize = 15)
plt.ylabel("Mean SPEI per month", fontsize = 15)
plt.title("Global Historic Drought - Standardized Precipitation & Evapotranspiration Index", size = 20)


# In[9]:


#Load Worldwide Google Search Interest for Drought 
google_drought = pd.read_csv('data/drought_google_2004_2022.csv')
google_drought = google_drought['Category: All categories'][1:]


# In[10]:


df2 = spei_df.mean(axis=0)
plt.figure(figsize=(25,6))
df2[1238:].plot()
plt.xticks(size = 15)
plt.yticks(size = 15)
# plt.xlim(0,1440)
plt.grid()
plt.xlabel("Time", fontsize = 15)
plt.ylabel("Mean SPEI per month", fontsize = 15)
plt.title("Global Historic Drought - Standardized Precipitation & Evapotranspiration Index (2004+)", size = 20)


# In[11]:


#Relationship between Google SI on Drought and mean SPEI (Globally)
spei_mean_2004_array = np.array(df2[1238:])
d = np.array(google_drought[:204].astype(int))
pg.corr(spei_mean_2004_array, d)


# In[12]:


# Convert all column names to str and limit the size to 10 characters since the hh:mm:ss are all the same
spei_df.columns = spei_df.columns.astype(str)
spei_df.columns = spei_df.columns.str.slice(stop=10)


# In[13]:


# spei_df['2020-12-16']


# In[14]:


# Load data into a geopandas dataframe
gdf = gpd.GeoDataFrame(
    spei_df['1920-05-16'], geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])

# Create a map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(15, 10), color='white', edgecolor='black')

# Normalize the data
norm = Normalize(vmin=gdf['1920-05-16'].min(), vmax=gdf['1920-05-16'].max())

#cmap 
cmap = cm.get_cmap('gray')
cmap = ListedColormap(cmap(np.linspace(0.0, 1.0, 256)))
cmap.set_bad(color='white')

# Create a patch collection
patches = [Circle((x, y), radius=0.5) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
pc = PatchCollection(patches, facecolor=cmap(norm(gdf['1920-05-16'])), alpha=0.8, edgecolor='gray')


# Add the patch collection to the map
ax.add_collection(pc)
ax.grid()

# Show the map
# plt.title('SPEI Map 1920-05-16', size=50)
plt.show()


# In[15]:


#Make a list of dates to iterate over and generate maps
dates = spei_df.columns[1238:].to_list()

dates_1ml = spei_df.columns[1237:-1].to_list()
dates_2ml = spei_df.columns[1236:-2].to_list()
dates_3ml = spei_df.columns[1235:-3].to_list()
dates_4ml = spei_df.columns[1234:-4].to_list()
dates_5ml = spei_df.columns[1233:-5].to_list()
dates_6ml = spei_df.columns[1232:-6].to_list()



# In[ ]:


for date in dates:

    # Load data into a geopandas dataframe
    gdf = gpd.GeoDataFrame(
        spei_df[date], geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])

    # Create a map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(figsize=(15, 10), color='white', edgecolor='black')

    # Normalize the data
    norm = Normalize(vmin=gdf[date].min(), vmax=gdf[date].max())

    #cmap 
    cmap = cm.get_cmap('gray')
    cmap = ListedColormap(cmap(np.linspace(0.0, 1.0, 256)))
    cmap.set_bad(color='white')

    # Create a patch collection
    patches = [Circle((x, y), radius=0.5) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    pc = PatchCollection(patches, facecolor=cmap(norm(gdf[date])), alpha=0.8, edgecolor='gray')


    # Add the patch collection to the map
    ax.add_collection(pc)
    ax.grid()

     # Save the map as a file
    ax.get_figure().savefig(f'globe_maps_spei/{date}_map.png')
    
    # Close the figure
    ax.get_figure().clf()


# In[34]:


# Load data into a geopandas dataframe
gdf = gpd.GeoDataFrame(spei_df, geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])

# Load the US states shapefile
us_map = gpd.read_file('cb_2018_us_state_500k.shp')

# Set the CRS of gdf to match us_map
gdf.set_crs(us_map.crs, inplace=True)

# Clip the points to the US
gdf = gpd.clip(gdf, us_map)

# Normalize the data
norm = Normalize(vmin=gdf['2020-09-16'].min(), vmax=gdf['2020-09-16'].max())

# cmap
cmap = plt.get_cmap('gray')
cmap = ListedColormap(cmap(np.linspace(0.0, 1.0, 256)))
cmap.set_bad(color='white')

# # Create a patch collection
# patches = [Circle((x, y), radius=0.3) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
# pc = PatchCollection(patches, facecolor=cmap(norm(gdf['1920-05-16'])), alpha=0.8, edgecolor='gray')
# Create a patch collection
square_size = 0.5
patches = [Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
pc = PatchCollection(patches, facecolor=cmap(norm(gdf['2020-09-16'])), alpha=0.8, edgecolor='gray')

# Plot the US states
ax = us_map.plot(figsize=(15, 10), color='white', edgecolor='black')

# Add the patch collection to the map
ax.add_collection(pc)

# Plot the US state borders on top
us_map.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

# Adjust plot limits to focus on the continental US
ax.set_xlim(-130, -65)
ax.set_ylim(24, 50)

# Show the map
plt.show()


# In[37]:


for date in dates_1ml:
    # Load data into a geopandas dataframe
    gdf = gpd.GeoDataFrame(spei_df, geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])

    # Load the US states shapefile
    us_map = gpd.read_file('cb_2018_us_state_500k.shp')

    # Set the CRS of gdf to match us_map
    gdf.set_crs(us_map.crs, inplace=True)

    # Clip the points to the US
    gdf = gpd.clip(gdf, us_map)

    # Normalize the data
    norm = Normalize(vmin=gdf[date].min(), vmax=gdf[date].max())

    # cmap
    cmap = plt.get_cmap('gray')
    cmap = ListedColormap(cmap(np.linspace(0.0, 1.0, 256)))
    cmap.set_bad(color='white')

    # # Create a patch collection
    # patches = [Circle((x, y), radius=0.3) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    # pc = PatchCollection(patches, facecolor=cmap(norm(gdf['1920-05-16'])), alpha=0.8, edgecolor='gray')
    # Create a patch collection
    square_size = 0.5
    patches = [Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    pc = PatchCollection(patches, facecolor=cmap(norm(gdf[date])), alpha=0.8, edgecolor='gray')

    # Plot the US states
    ax = us_map.plot(figsize=(15, 10), color='white', edgecolor='black')

    # Add the patch collection to the map
    ax.add_collection(pc)

    # Plot the US state borders on top
    us_map.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

    # Adjust plot limits to focus on the continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)

    # Save the map as a file
    ax.get_figure().savefig(f'usa_maps_spei_1ml/{date}_map.png')
    
    # Close the figure
    ax.get_figure().clf()
    


# In[38]:


for date in dates_2ml:
    # Load data into a geopandas dataframe
    gdf = gpd.GeoDataFrame(spei_df, geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])

    # Load the US states shapefile
    us_map = gpd.read_file('cb_2018_us_state_500k.shp')

    # Set the CRS of gdf to match us_map
    gdf.set_crs(us_map.crs, inplace=True)

    # Clip the points to the US
    gdf = gpd.clip(gdf, us_map)

    # Normalize the data
    norm = Normalize(vmin=gdf[date].min(), vmax=gdf[date].max())

    # cmap
    cmap = plt.get_cmap('gray')
    cmap = ListedColormap(cmap(np.linspace(0.0, 1.0, 256)))
    cmap.set_bad(color='white')

    # # Create a patch collection
    # patches = [Circle((x, y), radius=0.3) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    # pc = PatchCollection(patches, facecolor=cmap(norm(gdf['1920-05-16'])), alpha=0.8, edgecolor='gray')
    # Create a patch collection
    square_size = 0.5
    patches = [Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    pc = PatchCollection(patches, facecolor=cmap(norm(gdf[date])), alpha=0.8, edgecolor='gray')

    # Plot the US states
    ax = us_map.plot(figsize=(15, 10), color='white', edgecolor='black')

    # Add the patch collection to the map
    ax.add_collection(pc)

    # Plot the US state borders on top
    us_map.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

    # Adjust plot limits to focus on the continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)

    # Save the map as a file
    ax.get_figure().savefig(f'usa_maps_spei_2ml/{date}_map.png')
    
    # Close the figure
    ax.get_figure().clf()


# In[41]:


for date in dates_3ml:
    # Load data into a geopandas dataframe
    gdf = gpd.GeoDataFrame(spei_df, geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])

    # Load the US states shapefile
    us_map = gpd.read_file('cb_2018_us_state_500k.shp')

    # Set the CRS of gdf to match us_map
    gdf.set_crs(us_map.crs, inplace=True)

    # Clip the points to the US
    gdf = gpd.clip(gdf, us_map)

    # Normalize the data
    norm = Normalize(vmin=gdf[date].min(), vmax=gdf[date].max())

    # cmap
    cmap = plt.get_cmap('gray')
    cmap = ListedColormap(cmap(np.linspace(0.0, 1.0, 256)))
    cmap.set_bad(color='white')

    # # Create a patch collection
    # patches = [Circle((x, y), radius=0.3) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    # pc = PatchCollection(patches, facecolor=cmap(norm(gdf['1920-05-16'])), alpha=0.8, edgecolor='gray')
    # Create a patch collection
    square_size = 0.5
    patches = [Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    pc = PatchCollection(patches, facecolor=cmap(norm(gdf[date])), alpha=0.8, edgecolor='gray')

    # Plot the US states
    ax = us_map.plot(figsize=(15, 10), color='white', edgecolor='black')

    # Add the patch collection to the map
    ax.add_collection(pc)

    # Plot the US state borders on top
    us_map.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

    # Adjust plot limits to focus on the continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)

    # Save the map as a file
    ax.get_figure().savefig(f'usa_maps_spei_3ml/{date}_map.png')
    
    # Close the figure
    ax.get_figure().clf()


# In[42]:


for date in dates_4ml:
    # Load data into a geopandas dataframe
    gdf = gpd.GeoDataFrame(spei_df, geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])

    # Load the US states shapefile
    us_map = gpd.read_file('cb_2018_us_state_500k.shp')

    # Set the CRS of gdf to match us_map
    gdf.set_crs(us_map.crs, inplace=True)

    # Clip the points to the US
    gdf = gpd.clip(gdf, us_map)

    # Normalize the data
    norm = Normalize(vmin=gdf[date].min(), vmax=gdf[date].max())

    # cmap
    cmap = plt.get_cmap('gray')
    cmap = ListedColormap(cmap(np.linspace(0.0, 1.0, 256)))
    cmap.set_bad(color='white')

    # # Create a patch collection
    # patches = [Circle((x, y), radius=0.3) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    # pc = PatchCollection(patches, facecolor=cmap(norm(gdf['1920-05-16'])), alpha=0.8, edgecolor='gray')
    # Create a patch collection
    square_size = 0.5
    patches = [Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    pc = PatchCollection(patches, facecolor=cmap(norm(gdf[date])), alpha=0.8, edgecolor='gray')

    # Plot the US states
    ax = us_map.plot(figsize=(15, 10), color='white', edgecolor='black')

    # Add the patch collection to the map
    ax.add_collection(pc)

    # Plot the US state borders on top
    us_map.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

    # Adjust plot limits to focus on the continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)

    # Save the map as a file
    ax.get_figure().savefig(f'usa_maps_spei_4ml/{date}_map.png')
    
    # Close the figure
    ax.get_figure().clf()


# In[43]:


for date in dates_5ml:
    # Load data into a geopandas dataframe
    gdf = gpd.GeoDataFrame(spei_df, geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])

    # Load the US states shapefile
    us_map = gpd.read_file('cb_2018_us_state_500k.shp')

    # Set the CRS of gdf to match us_map
    gdf.set_crs(us_map.crs, inplace=True)

    # Clip the points to the US
    gdf = gpd.clip(gdf, us_map)

    # Normalize the data
    norm = Normalize(vmin=gdf[date].min(), vmax=gdf[date].max())

    # cmap
    cmap = plt.get_cmap('gray')
    cmap = ListedColormap(cmap(np.linspace(0.0, 1.0, 256)))
    cmap.set_bad(color='white')

    # # Create a patch collection
    # patches = [Circle((x, y), radius=0.3) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    # pc = PatchCollection(patches, facecolor=cmap(norm(gdf['1920-05-16'])), alpha=0.8, edgecolor='gray')
    # Create a patch collection
    square_size = 0.5
    patches = [Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    pc = PatchCollection(patches, facecolor=cmap(norm(gdf[date])), alpha=0.8, edgecolor='gray')

    # Plot the US states
    ax = us_map.plot(figsize=(15, 10), color='white', edgecolor='black')

    # Add the patch collection to the map
    ax.add_collection(pc)

    # Plot the US state borders on top
    us_map.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

    # Adjust plot limits to focus on the continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)

    # Save the map as a file
    ax.get_figure().savefig(f'usa_maps_spei_5ml/{date}_map.png')
    
    # Close the figure
    ax.get_figure().clf()





