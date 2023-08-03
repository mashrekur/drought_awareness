#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from IPython.display import Image
from datetime import datetime
import geopandas as gpd
from geopandas.tools import sjoin


# In[2]:


# Open the NETCDF file using xarray
ds = xr.open_dataset('data/spei01.nc')

# Convert the xarray dataset to a pandas dataframe
df = ds.to_dataframe()

df = df.unstack()

spei_df = df.dropna()
spei_df = spei_df['spei']

# fill the missing values lon in each column with the forward fill method.

spei_df = spei_df.reset_index()

spei_df = spei_df.apply(lambda x: x.ffill())

# Convert all column names to str and limit the size to 10 characters since the hh:mm:ss are all the same
spei_df.columns = spei_df.columns.astype(str)
spei_df.columns = spei_df.columns.str.slice(stop=10)


# In[21]:


#Make a list of dates to iterate over and generate maps
dates = spei_df.columns[2:].to_list()
dates


# In[30]:


spei_df.columns


# In[22]:


# Load data into a geopandas dataframe
gdf = gpd.GeoDataFrame(spei_df['2020-12-16'], geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])


# In[23]:


# Load the US states shapefile
us_map = gpd.read_file('cb_2018_us_state_500k.shp')


# In[34]:


# Function to calculate the average SPEI for the given time period
def calculate_avg_spei(spei_df, start_date, end_date):
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    
    total_months = (end_date_obj.year - start_date_obj.year) * 12 + (end_date_obj.month - start_date_obj.month)
    
    date_columns = [col for col in spei_df.columns[2:] if start_date <= col <= end_date]
    spei_df_filtered = spei_df[date_columns]
    
    avg_spei = spei_df_filtered.mean(axis=1).reset_index()
    avg_spei.columns = ['index', 'spei']
    avg_spei = pd.concat([spei_df[['lat', 'lon']], avg_spei], axis=1)
    
    return avg_spei


# In[41]:


def spei_to_geodataframe(avg_spei):
    # Create a GeoDataFrame from the avg_spei DataFrame
    avg_spei_gdf = gpd.GeoDataFrame(
        avg_spei, geometry=gpd.points_from_xy(avg_spei['lon'], avg_spei['lat'])
    )
    # Set the coordinate reference system (CRS) of the GeoDataFrame to match the us_map CRS
    avg_spei_gdf.crs = us_map.crs
    return avg_spei_gdf


# In[54]:


def plot_hotspots(gdf, avg_spei, period_years, output_file):
    # Convert the avg_spei DataFrame to a GeoDataFrame
    avg_spei_gdf = spei_to_geodataframe(avg_spei)
    # Perform a spatial join to keep only the data points that fall within the CONUS boundary
    avg_spei_within_us = sjoin(avg_spei_gdf, us_map, op='within')

    # Normalize the spei values for colormap
    norm = Normalize(vmin=avg_spei_within_us['spei'].min(), vmax=avg_spei_within_us['spei'].max())
    # Choose the reversed 'coolwarm' colormap
    cmap = plt.get_cmap('coolwarm_r')
    # Set the size of the squares for the patch collection
    square_size = 0.5
    # Create a list of Rectangle patches for the data points
    patches = [Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size) for x, y in zip(avg_spei_within_us.geometry.x, avg_spei_within_us.geometry.y)]
    # Create a patch collection with the patches and the colormap
    pc = PatchCollection(patches, facecolor=cmap(norm(avg_spei_within_us['spei'])), alpha=0.8, edgecolor='gray')

    # Plot the US states with white fill and black edges
    ax = us_map.plot(figsize=(15, 10), color='white', edgecolor='black')
    # Add the patch collection to the map
    ax.add_collection(pc)
    # Plot the US state borders on top
    us_map.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

    # Set the x and y limits to focus on the continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)
    # Add a title to the plot
    ax.set_title(f'Meteorological Drought Distribution for Past {period_years} Years (2004-2020)')

    # Create a scalar mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Add a colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax)
    # Set the label for the colorbar
    cbar.set_label('Average SPEI')

    # Save the plot to a file
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    # Show the plot
    plt.show()


# In[52]:


# Define the time periods to analyze
time_periods = [
    {"start_date": "1990-01-01", "end_date": "2020-12-31", "period_years": 30},
    {"start_date": "2000-01-01", "end_date": "2020-12-31", "period_years": 20},
    {"start_date": "2010-01-01", "end_date": "2020-12-31", "period_years": 10},
    {"start_date": "2015-01-01", "end_date": "2020-12-31", "period_years": 5},
]

# Generate the hotspot maps for each time period
for idx, period in enumerate(time_periods):
    avg_spei = calculate_avg_spei(spei_df, period["start_date"], period["end_date"])
    output_file = f"hotspots_map_{idx + 1}.png"
    plot_hotspots(us_map, avg_spei, period["period_years"], output_file)


# In[55]:


# Define the time periods to analyze
time_periods = [
    {"start_date": "2004-01-01", "end_date": "2020-12-31", "period_years": 16},
#     {"start_date": "2000-01-01", "end_date": "2020-12-31", "period_years": 20},
#     {"start_date": "2010-01-01", "end_date": "2020-12-31", "period_years": 10},
#     {"start_date": "2015-01-01", "end_date": "2020-12-31", "period_years": 5},
]

# Generate the hotspot maps for each time period
for idx, period in enumerate(time_periods):
    avg_spei = calculate_avg_spei(spei_df, period["start_date"], period["end_date"])
    output_file = f"hotspots_map_{idx + 1}.png"
    plot_hotspots(us_map, avg_spei, period["period_years"], output_file)


# In[ ]:




