#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry
import numpy as np
import rasterio
from rasterio.features import geometry_mask
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import json
from scipy.stats import linregress
import pymannkendall as mk
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from geopandas.tools import sjoin
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from adjustText import adjust_text
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter


# In[2]:


#Load state wise google SI 
usa_df = pd.read_csv('us_search_interest.csv')
# usa_df


# In[3]:


# Convert the 'Month' column to datetime 
usa_df['Month'] = pd.to_datetime(usa_df['Month'])


# In[4]:


# Cut off the dataframe after December 2020
usa_df = usa_df[usa_df['Month'] <= '2020-12']
# List of non-CONUS state names to exclude (adjust as necessary if using abbreviations)
non_conus_states = ['Alaska', 'Hawaii', 'Puerto Rico', 'United States Virgin Islands', 'District of Columbia']

# Filter out non-CONUS states and DC
usa_df = usa_df[~usa_df['Region'].isin(non_conus_states)]


# In[5]:


# Calculate the overall average Search Interest per state
average_si_per_state = usa_df.groupby('Region')['Search Interest'].mean().reset_index()


# In[105]:


usa_df['Month'] = pd.to_datetime(usa_df['Month'])
mean_search_interest = usa_df.groupby('Month')['Search Interest'].mean().reset_index()

# Step 4: Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(mean_search_interest['Month'], mean_search_interest['Search Interest'], marker='o', linestyle='-')
plt.title('Mean Search Interest Over Time')
plt.xlabel('Month')
plt.ylabel('Mean Search Interest')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()


# In[6]:


average_si_per_state


# In[7]:


# Load the US states shapefile
us_states = gpd.read_file('cb_2018_us_state_500k.shp')


# In[8]:


merged_df = us_states.merge(average_si_per_state, left_on='NAME', right_on='Region', how='right')
# merged_df

merged_df['Search Interest'] = merged_df['Search Interest'].fillna(0)


# Calculate the centroid of each geometry
merged_df['centroid'] = merged_df.geometry.centroid

merged_avg_search_interest = merged_df

# Increase the base font size
plt.rcParams.update({'font.size': 24})

# Plotting the map with figsize set for better visibility
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged_df.plot(column='Search Interest', cmap='Reds',ax=ax, legend=True,
               legend_kwds={})

# Add state abbreviations as annotations on the map
for idx, row in merged_df.iterrows():
    # Some geometries might be MultiPolygons, so we use representative_point() to get a good point for labeling
    point = row['geometry'].representative_point().coords[:][0]
    plt.annotate(text=row['STUSPS'], xy=point, horizontalalignment='center', fontsize=10)

# Adjust plot limits to focus on the continental US
ax.set_xlim(-130, -65)
ax.set_ylim(24, 50)
ax.set_xticks([])
ax.set_yticks([])

# Remove borders
for spine in ax.spines.values():
    spine.set_visible(False)
# ax.set_xlabel('Longitude', fontsize=14)
# ax.set_ylabel('Latitude', fontsize=14)
ax.set_title('US State-Wise Average Search Interest (2004-2020)', fontsize=24)

# Save the plot
plt.savefig('p1_overall_average_search_interest_map.png', bbox_inches='tight')
# Show the plot
plt.show()


# In[9]:


# Create a numerical representation of 'Month' for regression analysis (e.g., month number starting from 0)
usa_df['Month_Num'] = (usa_df['Month'] - usa_df['Month'].min()).dt.days / 30

# Calculate the trend (slope of the linear regression line) for each state
def calculate_trend(group):
    slope, intercept, r_value, p_value, std_err = linregress(group['Month_Num'], group['Search Interest'])
    return slope

trends = usa_df.groupby('Region').apply(calculate_trend).reset_index(name='Trend')

# Merge the trends data with the US states shapefile
merged_df = us_states.merge(trends, left_on='NAME', right_on='Region', how='right')

# Replace missing values with 0 for Trend
merged_df['Trend'] = merged_df['Trend'].fillna(0)
# Calculate the centroid of each geometry for annotation placement
merged_df['centroid'] = merged_df.geometry.centroid

merged_trend_search_interest = merged_df

# Plotting the map with trends and adding state abbreviations
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged_df.plot(column='Trend', cmap='Reds', ax=ax, legend=True,
               legend_kwds={})

# Add state abbreviations as annotations on the map
for idx, row in merged_df.iterrows():
    # Some geometries might be MultiPolygons, so we use representative_point() to get a good point for labeling
    point = row['centroid'].coords[:][0]
    plt.annotate(text=row['STUSPS'], xy=point, horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')

# Adjust plot limits to focus on the continental US
ax.set_xlim(-130, -65)
ax.set_ylim(24, 50)
ax.set_xticks([])
ax.set_yticks([])

# Remove borders
for spine in ax.spines.values():
    spine.set_visible(False)
# ax.set_xlabel('Longitude', fontsize=14)
# ax.set_ylabel('Latitude', fontsize=14)
ax.set_title('Trend of US State-Wise Search Interest Over Time', fontsize=24)

# Save the plot
plt.savefig('p1_search_interest_trend_map_with_state_names.png', bbox_inches='tight')
# Show the plot
plt.show()


# In[10]:


# Calculate the variance of search interest for each state
variance_si_per_state = usa_df.groupby('Region')['Search Interest'].var().reset_index(name='Variance')

# Merge the variance data with the US states shapefile
merged_df = us_states.merge(variance_si_per_state, left_on='NAME', right_on='Region', how='right')

# Replace missing values with 0 for Variance
merged_df['Variance'] = merged_df['Variance'].fillna(0)

# Calculate the centroid of each geometry for annotation placement
merged_df['centroid'] = merged_df.geometry.centroid

merged_var_search_interest = merged_df

# Plotting the map with variance and adding state abbreviations
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged_df.plot(column='Variance', cmap='Reds', ax=ax, legend=True,
               legend_kwds={})

# Add state abbreviations as annotations on the map
for idx, row in merged_df.iterrows():
    point = row['centroid'].coords[:][0]
    plt.annotate(text=row['STUSPS'], xy=point, horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')

# Adjust plot limits to focus on the continental US
ax.set_xlim(-130, -65)
ax.set_ylim(24, 50)
ax.set_xticks([])
ax.set_yticks([])

# Remove borders
for spine in ax.spines.values():
    spine.set_visible(False)
# ax.set_xlabel('Longitude', fontsize=14)
# ax.set_ylabel('Latitude', fontsize=14)
ax.set_title('Variance of US State-Wise Search Interest Over Time', fontsize=24)

# Save the plot
plt.savefig('p1_search_interest_variance_map_with_state_names.png', bbox_inches='tight')
# Show the plot
plt.show()


# In[11]:


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


# In[12]:


#Make a list of dates to iterate over and generate maps
dates = spei_df.columns[2:].to_list()


# Load data into a geopandas dataframe
gdf = gpd.GeoDataFrame(spei_df['2020-12-16'], geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])


# Load the US states shapefile
us_map = gpd.read_file('cb_2018_us_state_500k.shp')

# Filter out non-CONUS states from us_map
us_map = us_map[~us_map['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico', 'United States Virgin Islands'])]

# Correctly filter spei_df to retain 'lat', 'lon', and dates within 2004-2020
columns_to_keep = ['lat', 'lon'] + [col for col in spei_df.columns if '2004-01-16' <= col <= '2020-12-16']
spei_df_filtered = spei_df[columns_to_keep]

# Create a GeoDataFrame from the filtered SPEI DataFrame
gdf_spei = gpd.GeoDataFrame(
    spei_df_filtered,
    geometry=gpd.points_from_xy(spei_df_filtered.lon, spei_df_filtered.lat)
)
gdf_spei.crs = "EPSG:4326"  # Set the coordinate reference system to WGS84


spei_states = sjoin(gdf_spei, us_map, how="inner", op='intersects')

# Drop non-SPEI columns to focus on SPEI values
spei_values_columns = [col for col in spei_states.columns if '2004-01-16' <= col <= '2020-12-16']

# Calculate the mean SPEI for each row (pixel) across all time steps
spei_states['avg_spei'] = spei_states[spei_values_columns].mean(axis=1)

# Group by state NAME and calculate the average SPEI per state
state_avg_spei = spei_states.groupby('NAME')['avg_spei'].mean().reset_index()


# In[13]:


# Merge the average SPEI data with the US map GeoDataFrame
merged_avg_spei = us_map.merge(state_avg_spei, on='NAME')
merged_avg_spei['centroid'] = merged_avg_spei.geometry.centroid

# Plotting
fig, ax = plt.subplots(1, figsize=(15, 10))
merged_avg_spei.plot(column='avg_spei', cmap='RdYlGn', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Average SPEI per State (2004-2020)', fontsize=24)
ax.axis('off')

# Add state abbreviations as annotations
for idx, row in merged_avg_spei.iterrows():
    plt.annotate(s=row['STUSPS'], xy=(row['centroid'].x, row['centroid'].y),
                 horizontalalignment='center', fontsize=10, color='black')

# Save the plot
plt.savefig('p1_average_spei_per_state.png', bbox_inches='tight')
plt.show()


# In[14]:


# Convert to long format
spei_long = spei_df_filtered.melt(id_vars=["lat", "lon"], var_name="date", value_name="SPEI")
spei_long['date'] = pd.to_datetime(spei_long['date'])

# Convert dates to a numerical format for regression analysis
spei_long['time'] = spei_long['date'].apply(lambda x: x.toordinal())

# Convert to GeoDataFrame
gdf_spei = gpd.GeoDataFrame(
    spei_long,
    geometry=[Point(xy) for xy in zip(spei_long.lon, spei_long.lat)],
    crs="EPSG:4326"
)

# Spatial join SPEI trends with US states
spei_states = sjoin(gdf_spei, us_map, how="inner", op='intersects')

# Group by state and date to calculate the average SPEI per state for each time step
avg_spei_per_state_time = spei_states.groupby(['NAME', 'date'])['SPEI'].mean().reset_index()

# Prepare the data for trend calculation
avg_spei_per_state_time['time'] = avg_spei_per_state_time['date'].apply(lambda x: x.toordinal())

# Calculate trend for each state
def calculate_state_trend(group):
    slope, intercept, r_value, p_value, std_err = linregress(group['time'], group['SPEI'])
    return pd.Series({'slope': slope})

state_trends = avg_spei_per_state_time.groupby('NAME').apply(calculate_state_trend).reset_index()

# Merge the trend data with the US map GeoDataFrame
merged_trend_spei = us_map.merge(state_trends, left_on='NAME', right_on='NAME')
merged_trend_spei['centroid'] = merged_trend_spei.geometry.centroid

# Plotting the Trends in SPEI per State (2004-2020)
fig, ax = plt.subplots(1, figsize=(15, 10))
merged_trend_spei.plot(column='slope', cmap= 'RdYlGn', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Trends in SPEI per State (2004-2020)', fontsize=24)
ax.axis('off')

# Add state abbreviations as annotations
for idx, row in merged_trend_spei.iterrows():
    plt.annotate(s=row['STUSPS'], xy=(row['centroid'].x, row['centroid'].y),
                 horizontalalignment='center', fontsize=10, color='black' if row['slope'] < 0 else 'black')
# Save the plot
plt.savefig('p1_trends_spei_per_state.png', bbox_inches='tight')
plt.show()


# In[15]:


# Calculate variance of SPEI for each state and date
spei_variance = spei_states.groupby(['NAME', 'date'])['SPEI'].var().reset_index(name='variance')

# Calculate the average variance of SPEI per state over the period
state_variance_spei = spei_variance.groupby('NAME')['variance'].mean().reset_index()

# Merge the variance data with the US map GeoDataFrame
merged_variance_spei = us_map.merge(state_variance_spei, on='NAME')
merged_variance_spei['centroid'] = merged_variance_spei.geometry.centroid

# Plotting the Variance in SPEI per State (2004-2020)
fig, ax = plt.subplots(1, figsize=(15, 10))
merged_variance_spei.plot(column='variance', cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Variance in SPEI per State (2004-2020)', fontsize=24)
ax.axis('off')

# Add state abbreviations as annotations
for idx, row in merged_variance_spei.iterrows():
    plt.annotate(s=row['STUSPS'], xy=(row['centroid'].x, row['centroid'].y),
                 horizontalalignment='center', fontsize=10, color='black')
# Save the plot
plt.savefig('p1_variance_spei_per_state.png', bbox_inches='tight')
plt.show()


# In[68]:


common_states = set(merged_avg_search_interest['STUSPS']) & set(merged_avg_spei['STUSPS'])

merged_avg_search_interest_aligned = merged_avg_search_interest[merged_avg_search_interest['STUSPS'].isin(common_states)].sort_values('STUSPS')
merged_avg_spei_aligned = merged_avg_spei[merged_avg_spei['STUSPS'].isin(common_states)].sort_values('STUSPS')

def plot_save_correlation(y, x, state_abbrs, y_label, x_label, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y, x, s=5, color='blue')  # Adjust dot size for visibility
    
    offsets = {'NC': (-0.01, 0.0), 'FL': (-0.01, 0.0), 'TX': (-0.01, 0.0), 'MN': (-0.01, 0.0), 'PA': (-0.01, 0.0)}
    
    for i, txt in enumerate(state_abbrs):
        offset = offsets.get(txt, (0, 0))
        ax.annotate(txt, (y.iloc[i] + offset[0], x.iloc[i] + offset[1]), fontsize=10, ha='center')
    
    # Calculate Pearson correlation coefficient and p-value
    corr_coef, p_value = pearsonr(x, y)
    ax.set_title(f'{title}\nCorrelation: {corr_coef:.2f}, p-value: {p_value:.2e}')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    
    ax.grid(True)
    
    # Calculate and plot the best fit line
    m, b = np.polyfit(y, x, 1)
    ax.plot(y, m*y + b, color='red', label=f'Best fit: y={m:.2f}x+{b:.2f}')
    ax.legend(fontsize='small') 
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
plot_save_correlation(
    y=merged_avg_spei_aligned['avg_spei'],
    x=merged_avg_search_interest_aligned['Search Interest'],
    state_abbrs=merged_avg_search_interest_aligned['STUSPS'],
    y_label='Average Search Interest',
    x_label='Average SPEI',
    title='Average SPEI and Search Interest',
    save_path='p1_avg_spei_search_interest_correlation_states.png'
)


# In[69]:


common_states_trend = set(merged_trend_search_interest['STUSPS']) & set(merged_trend_spei['STUSPS'])

merged_trend_search_interest_aligned = merged_trend_search_interest[merged_trend_search_interest['STUSPS'].isin(common_states_trend)].sort_values('STUSPS')
merged_trend_spei_aligned = merged_trend_spei[merged_trend_spei['STUSPS'].isin(common_states_trend)].sort_values('STUSPS')

def plot_detailed_correlation(x, y, state_abbrs, x_label, y_label, title, save_path, excluded_states=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, s=5, color='blue')  # Adjust dot size for visibility
    
    if excluded_states is None:
        excluded_states = set()
    
    for i, txt in enumerate(state_abbrs):
        if txt not in excluded_states:
            ax.annotate(txt, (x.iloc[i], y.iloc[i]), fontsize=10, ha='center')
    
    # Calculate Pearson correlation coefficient and p-value
    corr_coef, p_value = pearsonr(x, y)
    
    # Adjust title to include p-value
    ax.set_title(f'{title}\nCorrelation: {corr_coef:.2f}, p-value: {p_value:.2e}')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Calculate and plot the best fit line
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color='red', label=f'Best fit: y={m:.2f}x+{b:.2f}')
    ax.legend(fontsize='small') 
    
    # Set the ScalarFormatter for the x-axis
    formatter = ScalarFormatter(useOffset=True, useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))  # Adjust the range for scientific notation
    ax.xaxis.set_major_formatter(formatter)
    
    ax.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage with the specified excluded states
excluded_states = {'UT', 'NY', 'MN', 'MO'}  # Example set of states to exclude from annotations

plot_detailed_correlation(
    x=merged_trend_spei_aligned['slope'], 
    y=merged_trend_search_interest_aligned['Trend'],  
    state_abbrs=merged_trend_search_interest_aligned['STUSPS'],  # State abbreviations
    x_label='Trend in SPEI',
    y_label='Trend in Search Interest',
    title='Trend in SPEI and Search Interest',
    save_path='p1_trend_spei_search_interest_correlation_excluded.png',
    excluded_states=excluded_states
)


# In[112]:


merged_trend_search_interest_aligned


# In[114]:


# # Calculate the correlation for each state
# state_correlations = []
# for state in merged_trend_search_interest_aligned['STUSPS'].unique():
#     spei = merged_trend_spei_aligned.loc[merged_trend_spei_aligned['STUSPS'] == state, 'slope']
#     print(spei)
#     search_interest = merged_trend_search_interest_aligned.loc[merged_trend_search_interest_aligned['STUSPS'] == state, 'Trend']
#     print(search_interest)
#     if not spei.empty and not search_interest.empty:
#         correlation, _ = pearsonr(spei, search_interest)
#         state_correlations.append({'STUSPS': state, 'Correlation': correlation})

# # Convert to DataFrame
# correlation_df = pd.DataFrame(state_correlations)

# # Merge the correlation data back into the geodataframe for plotting
# gdf = gpd.GeoDataFrame(merged_trend_search_interest_aligned, geometry='geometry')
# gdf = gdf.merge(correlation_df, on='STUSPS', how='left')

# # Plotting
# fig, ax = plt.subplots(1, figsize=(15, 10))
# gdf.plot(column='Correlation', cmap='coolwarm', ax=ax, legend=True,
#          legend_kwds={'label': "Correlation between SPEI slope and Search Interest Trend", 'orientation': "horizontal"})
# ax.set_title('State-wise Correlation between SPEI Slope and Search Interest Trend', fontsize=14)

# # Annotating state abbreviations
# for idx, row in gdf.iterrows():
#     ax.annotate(text=row['STUSPS'], xy=(row.geometry.centroid.x, row.geometry.centroid.y), ha='center', fontsize=6)

# ax.set_axis_off()
# plt.savefig('p2_trends_conus_map.png', dpi=300, bbox_inches='tight')
# plt.show()


# In[70]:


common_states_variance = set(merged_var_search_interest['STUSPS']) & set(merged_variance_spei['STUSPS'])

merged_var_search_interest_aligned = merged_var_search_interest[merged_var_search_interest['STUSPS'].isin(common_states_variance)].sort_values('STUSPS')
merged_var_spei_aligned = merged_variance_spei[merged_variance_spei['STUSPS'].isin(common_states_variance)].sort_values('STUSPS')

# Specify which states to exclude from annotations
excluded_states_variance = {'NV'}  

def plot_detailed_correlation_variance(x, y, state_abbrs, x_label, y_label, title, save_path, excluded_states=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Ensure x and y are pandas Series to simplify handling
    x, y = pd.Series(x), pd.Series(y)
    
    # Remove rows with NaN or inf values in either x or y
    clean_indices = np.isfinite(x) & np.isfinite(y)
    x_clean = x[clean_indices]
    y_clean = y[clean_indices]
    state_abbrs_clean = state_abbrs[clean_indices]
    
    ax.scatter(x_clean, y_clean, s=5, color='blue')  # Adjusted dot size for visibility
    
    if excluded_states is None:
        excluded_states = set()
    
    for i, txt in enumerate(state_abbrs_clean):
        if txt not in excluded_states:
            ax.annotate(txt, (x_clean.iloc[i], y_clean.iloc[i]), fontsize=10, ha='center')
    
    # Calculate Pearson correlation coefficient and p-value with cleaned data
    corr_coef, p_value = pearsonr(x_clean, y_clean)
    
    # Update title to include p-value
    ax.set_title(f'{title}\nCorrelation: {corr_coef:.2f}, p-value: {p_value:.2e}')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Calculate and plot the best fit line
    m, b = np.polyfit(x_clean, y_clean, 1)
    ax.plot(x_clean, m*x_clean + b, color='red', label=f'Best fit: y={m:.2f}x+{b:.2f}')
    ax.legend(fontsize='small') 
    
    ax.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_detailed_correlation_variance(
    x=merged_var_spei_aligned['variance'],  
    y=merged_var_search_interest_aligned['Variance'],  
    state_abbrs=merged_var_search_interest_aligned['STUSPS'],  # State abbreviations
    x_label='Variance in SPEI',
    y_label='Variance in Search Interest',
    title='Variance in SPEI and Search Interest',
    save_path='p1_variance_correlation_spei_si.png',
    excluded_states=excluded_states_variance
)


# In[19]:


#Make a list of dates to iterate over and generate maps
dates = spei_df.columns[2:].to_list()


# Load data into a geopandas dataframe
gdf = gpd.GeoDataFrame(spei_df['2020-12-16'], geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])


# Load the US states shapefile
us_map = gpd.read_file('cb_2018_us_state_500k.shp')



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


def spei_to_geodataframe(avg_spei):
    # Create a GeoDataFrame from the avg_spei DataFrame
    avg_spei_gdf = gpd.GeoDataFrame(
        avg_spei, geometry=gpd.points_from_xy(avg_spei['lon'], avg_spei['lat'])
    )
    # Set the coordinate reference system (CRS) of the GeoDataFrame to match the us_map CRS
    avg_spei_gdf.crs = us_map.crs
    return avg_spei_gdf


# In[20]:


# Filter for the start and end months
start_month_interest = usa_df[usa_df['Month'] == '2004-01'].groupby('Region')['Search Interest'].mean()
end_month_interest = usa_df[usa_df['Month'] == '2020-12'].groupby('Region')['Search Interest'].mean()

# Calculate the rise in search interest
rise_in_interest = end_month_interest - start_month_interest
rise_in_interest_df = rise_in_interest.reset_index().rename(columns={'Search Interest': 'Rise in Interest'})


# In[21]:


# Load the saved R-squared array
r_squared_array = np.load("output/r_squared_array.npy", allow_pickle=True)


# In[22]:


r_squared_array.shape[0]/6



num_models = 6

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(25, 30))  # Adjust the size of the figure
axs = axs.flatten()  # Flatten the array of axes for easy indexing

# Define major interval for the grid
major_interval = 50

for model_index in range(num_models):
    one_model_results = r_squared_array[model_index::num_models]

    # Initialize an array for the cropped R-squared values, fill with NaN for missing data
    cropped_r_squared_values = np.full((lower_bound - upper_bound, right_bound - left_bound), np.nan)

    # Assign the R-squared values to the corresponding pixels in the cropped array
    for ((row, col), r_squared) in one_model_results:
        if upper_bound <= row < lower_bound and left_bound <= col < right_bound:
            cropped_row_index = row - upper_bound
            cropped_col_index = col - left_bound
            cropped_r_squared_values[cropped_row_index, cropped_col_index] = max(r_squared, 0.16)  # Set values less than 0.1 to 0

    # Plot each model's data in a subplot
    ax = axs[model_index]
    pcm = ax.imshow(cropped_r_squared_values, cmap='Reds', interpolation='nearest', norm=colors.PowerNorm(gamma=0.5))
    ax.set_title(f'Model {model_index}ML', fontsize=20)

    # Add major grid lines
    ax.set_xticks(np.arange(0, cropped_r_squared_values.shape[1], major_interval))
    ax.set_yticks(np.arange(0, cropped_r_squared_values.shape[0], major_interval))
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)

    # Customize the tick labels to show whole numbers starting from zero
    ax.set_xticklabels(np.arange(0, cropped_r_squared_values.shape[1], major_interval), fontsize=15)
    ax.set_yticklabels(np.arange(0, cropped_r_squared_values.shape[0], major_interval), fontsize=15)

# Add a colorbar outside the area of the subplots
fig.subplots_adjust(right=1.25)
cbar_ax = fig.add_axes([1.30, 0.15, 0.02, 0.6])  # x, y, width, height (adjust as needed)


cbar = fig.colorbar(pcm, cax=cbar_ax, label='R-squared values')

cbar.set_label('R-squared values', fontsize=30)
# Save the figure
plt.savefig('r_squared_models_figure.png', bbox_inches='tight')


# Show the plot
plt.show()


# In[27]:


# Load state boundaries
with open('state_boundaries.json', 'r') as file:
    state_boundaries = json.load(file)

num_models = 6
state_averages_per_model = {}

# Calculate average R-squared for each state for each model
for model_index in range(num_models):
    cropped_r_squared_values = np.load(f'cropped_r_squared_{model_index}ML.npy')
    
    state_averages = {}
    for state, bounds in state_boundaries.items():
        nw, sw, ne, se = bounds['NW'], bounds['SW'], bounds['NE'], bounds['SE']
        state_pixels = cropped_r_squared_values[nw[0]:se[0], nw[1]:ne[1]]
        state_avg_r_squared = np.nanmean(state_pixels)
        state_averages[state] = state_avg_r_squared
    
    state_averages_per_model[model_index] = state_averages

# Save the average R-squared values per state per model
with open('state_averages_per_model.json', 'w') as file:
    json.dump(state_averages_per_model, file)


# In[28]:


# overall_state_averages = {state: np.mean([model_data[state] for model_data in state_averages_per_model.values()]) for state in state_boundaries.keys()}

# # Save the overall average R-squared values
# with open('overall_state_averages.json', 'w') as file:
#     json.dump(overall_state_averages, file)


# In[29]:


# Load the average R-squared values per state per model from JSON
with open('state_averages_per_model.json', 'r') as file:
    state_averages_per_model = json.load(file)

# Convert the loaded JSON data into a DataFrame
state_averages_df = pd.DataFrame(state_averages_per_model)  # Transpose to have states as rows and models as columns
state_averages_df.columns = [f'Model_{i}ML' for i in range(len(state_averages_df.columns))]  # Rename columns to 'Model_XML'


# In[30]:


state_averages_df


# In[31]:


# Merge the GeoDataFrame with the state averages DataFrame
avg_state_model_df = us_states.merge(state_averages_df, left_on='NAME', right_index=True)

# Prepare the figure for the subplots
fig, axs = plt.subplots(3, 2, figsize=(26, 17))  # Adjust the size 
axs = axs.flatten()

# Define the colormap
cmap = plt.cm.Reds

# Find global min and max R-squared values across all models for a shared colorbar
model_columns = ['Model_0ML', 'Model_1ML', 'Model_2ML', 'Model_3ML', 'Model_4ML', 'Model_5ML']
all_values = avg_state_model_df[model_columns].values.flatten()
vmin, vmax = all_values.min(), all_values.max()

for ax, model_column in zip(axs, model_columns):
    # Plot the map for each model
    avg_state_model_df.plot(column=model_column, cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', vmin=vmin, vmax=vmax)
    
    # Add state abbreviations as annotations on the map
    for idx, row in avg_state_model_df.iterrows():
        point = row['geometry'].representative_point().coords[:][0]
        ax.annotate(text=row['STUSPS'], xy=point, horizontalalignment='center', verticalalignment='center', fontsize=10)
    
    ax.set_title(model_column)
    ax.axis('off')

# Adjust subplot layout
plt.tight_layout()

# Add a shared colorbar to the right of the subplots
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # Adjust as needed for layout
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
fig.colorbar(sm, cax=cbar_ax)

# Set the title for the entire figure
fig.suptitle('Model R-squared Averaged Over States', fontsize=25)
fig.subplots_adjust(top=0.9)
# Save the figure
plt.savefig('p2_avg_r_squared_per_state_per_model.png', bbox_inches='tight')

plt.show()


# In[32]:


# Load the overall average R-squared values
with open('overall_state_averages.json', 'r') as file:
    overall_state_averages = json.load(file)

# Convert the loaded JSON data into a DataFrame
overall_state_averages_df = pd.DataFrame(list(overall_state_averages.items()), columns=['NAME', 'Overall_R_squared'])

# Merge the GeoDataFrame with the overall state averages DataFrame
overall_avg_state_model_df = us_states.merge(overall_state_averages_df, on='NAME')

# Calculate the centroid of each state's geometry for annotation
overall_avg_state_model_df['centroid'] = overall_avg_state_model_df.geometry.centroid

# Plotting the map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))  # Adjust the size as needed

# Define the colormap
cmap = plt.cm.Reds

# Plot the map
overall_avg_state_model_df.plot(column='Overall_R_squared', cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                               legend_kwds={'shrink': 0.5, 'orientation': "horizontal"})

# Add state abbreviations as annotations on the map
for idx, row in overall_avg_state_model_df.iterrows():
    point = row['centroid'].coords[:][0]  # Using centroids for labeling
    ax.annotate(text=row['STUSPS'], xy=point, horizontalalignment='center', verticalalignment='center', fontsize=10)

ax.set_title('Overall Model R-squared Averaged Over States')
ax.axis('off')
# Save the figure
plt.savefig('p3_avg_r_squared_per_state_overall.png', bbox_inches='tight')

plt.show()


# In[33]:


regions = {
    'Western': ['Arizona', 'California', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Oregon', 'Utah', 'Washington', 'Wyoming'],
    'Central': ['Alabama', 'Arkansas', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Nebraska', 'North Dakota', 'Ohio', 'Oklahoma', 'South Dakota', 'Tennessee', 'Texas', 'Wisconsin'],
    'Northeastern': ['Connecticut', 'Delaware', 'Maine', 'Maryland', 'Massachusetts', 'New Hampshire', 'New Jersey', 'New York', 'Pennsylvania', 'Rhode Island', 'Vermont', 'Virginia', 'West Virginia'],
    'Southeastern': ['Florida', 'Georgia', 'North Carolina', 'South Carolina']
}

# Create a DataFrame from the JSON data
state_r2_df = pd.DataFrame(list(overall_state_averages.items()), columns=['State', 'R_squared'])

# Map states to regions
state_r2_df['Region'] = state_r2_df['State'].apply(lambda x: next((region for region, states in regions.items() if x in states), None))

# Calculate average R-squared per region and sort
region_r2_avg = state_r2_df.groupby('Region')['R_squared'].mean().sort_values(ascending=False)

# Generate a color for each bar
colors = plt.cm.viridis(np.linspace(0, 1, len(region_r2_avg)))

# Plotting the bar chart with different colors
region_r2_avg.plot(kind='bar', figsize=(10, 6), color=colors)
plt.title('Average R-Squared Values by Region')
plt.ylabel('Average R-Squared')
plt.xlabel('Region')
plt.xticks(rotation=45)
# Save the figure
plt.savefig('avg_r_squared_per_region.png', bbox_inches='tight')
plt.show()


# In[34]:


us_states['Region'] = us_states['NAME'].apply(lambda x: next((region for region, states in regions.items() if x in states), None))

# Convert region_r2_avg to a dictionary for easier mapping
region_r2_avg_dict = region_r2_avg.to_dict()

# Map the average R-squared value of each region to each state
us_states['Avg_R_squared_by_Region'] = us_states['Region'].map(region_r2_avg_dict)

fig, ax = plt.subplots(1, figsize=(15, 10))  # Adjust the size as needed

# Plot the map
us_states.plot(column='Avg_R_squared_by_Region', cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
               legend_kwds={'shrink': 0.5, 'orientation': "horizontal"})

# Add state abbreviations as annotations on the map
for idx, row in us_states.iterrows():
    point = row['geometry'].centroid.coords[:][0]  # Using centroids for labeling
    ax.annotate(text=row['STUSPS'], xy=point, horizontalalignment='center', verticalalignment='center', fontsize=10)

ax.set_title('Overall Model R-squared Averaged by Region')
ax.axis('off')

# Save the figure
plt.savefig('p3_avg_r_squared_per_state_by_region.png', bbox_inches='tight')
plt.show()


# In[71]:


state_abbr_mapping = merged_avg_spei_aligned.set_index('NAME')['STUSPS'].to_dict()

# merged_df['STUSPS'] = merged_df['NAME'].map(state_abbr_mapping)

# Remove rows where 'STUSPS' is NaN
merged_df_r2 = merged_avg_search_interest.dropna(subset=['STUSPS'])

overall_r_squared_mapping = overall_state_averages_df.set_index('NAME')['Overall_R_squared'].to_dict()

# Map 'Overall_R_squared' to merged_df based on 'NAME'
merged_df_r2['Overall_R_squared'] = merged_df_r2['NAME'].map(overall_r_squared_mapping)


def plot_search_interest_vs_r_squared_correlation(x, y, state_abbrs, x_label, y_label, title, save_path, excluded_states=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Ensure x and y are pandas Series to simplify handling
    x, y = pd.Series(x), pd.Series(y)
    
    # Remove rows with NaN or inf values in either x or y
    clean_indices = np.isfinite(x) & np.isfinite(y)
    x_clean = x[clean_indices]
    y_clean = y[clean_indices]
    state_abbrs_clean = state_abbrs[clean_indices]
    
    ax.scatter(x_clean, y_clean, s=5, color='blue')  # Adjusted dot size for visibility
    
    if excluded_states is None:
        excluded_states = set()
    
    for i, txt in enumerate(state_abbrs_clean):
        if txt not in excluded_states:
            ax.annotate(txt, (x_clean.iloc[i], y_clean.iloc[i]), fontsize=10, ha='center')
    
    # Calculate Pearson correlation coefficient and p-value with cleaned data
    corr_coef, p_value = pearsonr(x_clean, y_clean)
    
    # Update title to include both correlation coefficient and p-value
    ax.set_title(f'{title}\nCorrelation: {corr_coef:.2f}, p-value: {p_value:.2e}')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Calculate and plot the best fit line
    m, b = np.polyfit(x_clean, y_clean, 1)
    ax.plot(x_clean, m*x_clean + b, color='red', label=f'Best fit: y={m:.2f}x+{b:.2f}')
    ax.legend(fontsize='small') 
    
    ax.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
# Specify which states to exclude from annotations
excluded_states = {'TX', 'FL', 'AL'}  

# Create the plot for search interest vs R-squared correlation with selective exclusion of annotations
plot_search_interest_vs_r_squared_correlation(
    x=merged_df_r2['Search Interest'],
    y=merged_df_r2['Overall_R_squared'],
    state_abbrs=merged_df_r2['STUSPS'],
    x_label='Search Interest',
    y_label='Overall R-squared',
    title='Search Interest and Overall R-squared',
    save_path='p4_search_interest_vs_r_squared_correlation.png',
    excluded_states=excluded_states  # Pass the filter for excluded states
)


# In[72]:


# plotting trends correlation
plot_search_interest_vs_r_squared_correlation(
    x=merged_trend_search_interest_aligned['Trend'],  
    y=merged_df_r2['Overall_R_squared'],  
    state_abbrs=merged_trend_search_interest_aligned['STUSPS'],  # State abbreviations
    x_label='Trend in Search Interest',
    y_label='Overall R-squared',
    title='Trend in Search Interest and R-squared',
    save_path='p4_trend_search_interest_vs_r_squared_correlation.png',
    excluded_states={'WV', 'AL', 'MS', 'GA', 'IN'}  
)


# In[73]:


# call for plotting variance correlation
plot_search_interest_vs_r_squared_correlation(
    x=merged_var_search_interest_aligned['Variance'],  
    y=merged_df_r2['Overall_R_squared'],  
    state_abbrs=merged_var_search_interest_aligned['STUSPS'],  # State abbreviations
    x_label='Variance in Search Interest',
    y_label='Overall R-squared',
    title='Variance in Search Interest and R-squared',
    save_path='p4_variance_search_interest_vs_r_squared_correlation.png',
    excluded_states={'MN', 'LA', 'SD', 'MN', 'MA'}  
)


# In[99]:


def plot_spei_vs_r_squared_correlation(x, y, state_abbrs, x_label, y_label, title, save_path, excluded_states=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot points
    ax.scatter(x, y, s=5, color='blue')  # Adjusted dot size for better visibility
    
    # Annotate points with state abbreviations, excluding specified states
    if excluded_states is None:
        excluded_states = set()
    for i, txt in enumerate(state_abbrs):
        if txt not in excluded_states:
            ax.annotate(txt, (x.iloc[i], y.iloc[i]), fontsize=10, ha='center')

    try:
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color='red', label=f'Best fit: y={m:.2f}x+{b:.2f}')
    except np.linalg.LinAlgError:
        print("Could not converge to a solution.")
        
     # Calculate and plot the best fit line

    ax.legend(fontsize='small') 

    
    # Calculate Pearson correlation coefficient and p-value
    corr_coef, p_value = pearsonr(x, y)
    
    # Set the ScalarFormatter for the x-axis
    formatter = ScalarFormatter(useOffset=True, useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))  # Adjust the range for scientific notation
    ax.xaxis.set_major_formatter(formatter)
    
    ax.set_title(f'{title}\nCorrelation: {corr_coef:.2f}, p-value: {p_value:.2e}')

    
    # This will add the common multiplier in the axis label
    ax.get_xaxis().get_offset_text().set_position((1,0))  # Adjust position as needed
    # Format x-axis labels to use scientific notation
#     ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
    
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    

    
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# In[100]:


plot_spei_vs_r_squared_correlation(
    x=merged_avg_spei_aligned['avg_spei'],
    y=merged_df_r2['Overall_R_squared'],
    state_abbrs=merged_avg_spei_aligned['STUSPS'],
    x_label='Average SPEI',
    y_label='Overall R-squared',
    title='Average SPEI and R-squared',
    save_path='p4_avg_spei_vs_r_squared_correlation.png',
    excluded_states={'ND'}  
)


# In[101]:


plot_spei_vs_r_squared_correlation(
    x=merged_trend_spei_aligned['slope'],
    y=merged_df_r2['Overall_R_squared'],
    state_abbrs=merged_trend_spei_aligned['STUSPS'],
    x_label='Trend in SPEI',
    y_label='Overall R-squared',
    title='Trend in SPEI and R-squared',
    save_path='p4_trend_spei_vs_r_squared_correlation.png',
    excluded_states={'MS'} 
)


# In[41]:


len(merged_var_spei_aligned['variance'])


# In[96]:


def plot_spei_vs_r_squared_correlation(x, y, state_abbrs, x_label, y_label, title, save_path, excluded_states=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Convert x, y, and state_abbrs to pandas Series to ensure alignment
    x = pd.Series(x)
    y = pd.Series(y)
    state_abbrs = pd.Series(state_abbrs.values, index=x.index)  # Ensure state_abbrs aligns with x and y
    
    # Filter out NaNs and Infs from x, y, and state_abbrs
    valid_indices = np.isfinite(x) & np.isfinite(y)
    x_clean = x[valid_indices]
    y_clean = y[valid_indices]
    state_abbrs_clean = state_abbrs[valid_indices]
    
    # Plotting
    ax.scatter(x_clean, y_clean, s=5, color='blue')  # Adjusted dot size for better visibility
    
    # Annotations, excluding specified states
    if excluded_states is None:
        excluded_states = set()
    for txt, xpos, ypos in zip(state_abbrs_clean, x_clean, y_clean):
        if txt not in excluded_states:
            ax.annotate(txt, (xpos, ypos), fontsize=10, ha='center')
    
    # Calculate and plot best fit line
    m, b = np.polyfit(x_clean, y_clean, 1)
    ax.plot(x_clean, m*x_clean + b, 'r-', label=f'Best fit: y={m:.2f}x+{b:.2f}')
    ax.legend(fontsize='small') 
    
    # Calculate Pearson correlation coefficient and p-value
    corr_coef, p_value = pearsonr(x_clean, y_clean)
    ax.set_title(f'{title}\nCorrelation: {corr_coef:.2f}, p-value: {p_value:.2e}')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


plot_spei_vs_r_squared_correlation(
    x=merged_var_spei_aligned['variance'],
    y=merged_df_r2['Overall_R_squared'],
    state_abbrs=merged_var_spei_aligned['STUSPS'],
    x_label='Variance in SPEI',
    y_label='Overall R-squared',
    title='Variance in SPEI and R-squared',
    save_path='p4_variance_spei_vs_r_squared_correlation.png',
    excluded_states={'WY', 'UT', 'SD', 'ME'}  
)


# In[131]:


# Define the figure paths in the order they'll be displayed
fig_paths = [
    'p1_overall_average_search_interest_map.png', 'p1_search_interest_trend_map_with_state_names.png',
    'p1_search_interest_variance_map_with_state_names.png', 'p1_average_spei_per_state.png',
    'p1_trends_spei_per_state.png', 'p1_variance_spei_per_state.png',
    'p1_avg_spei_search_interest_correlation_states.png', 'p1_trend_spei_search_interest_correlation_excluded.png',
    'p1_variance_correlation_spei_si.png'
]

# Labels for each subplot
subplot_labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)']

# Creating the panel with adjusted spacing
fig, axs = plt.subplots(3, 3, figsize=(30, 24))
plt.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust spacing

# Loop over each subplot, image path, and label
for ax, fig_path, label in zip(axs.ravel(), fig_paths, subplot_labels):
    img = mpimg.imread(fig_path)
    ax.imshow(img)
    ax.axis('off')
    # Position the label in the top-left corner of the subplot
    ax.text(0.0, 1.0, label, transform=ax.transAxes, fontsize=28, va='top', ha='left')

plt.tight_layout()
plt.savefig('panel_1.png', bbox_inches='tight')
plt.show()


# In[44]:


# # Load the overall average R-squared values
# with open('overall_state_averages.json', 'r') as file:
#     overall_state_averages = json.load(file)

# # Convert the overall state averages to a DataFrame
# overall_state_averages_df = pd.DataFrame(list(overall_state_averages.items()), columns=['State', 'Overall_R_squared'])

# # Load state wise google search interest data
# usa_df = pd.read_csv('us_search_interest.csv')
# usa_df['Month'] = pd.to_datetime(usa_df['Month'])
# usa_df = usa_df[usa_df['Month'] <= '2020-12']
# average_si_per_state = usa_df.groupby('Region')['Search Interest'].mean().reset_index()

# # Merge the two datasets
# merged_df = pd.merge(average_si_per_state, overall_state_averages_df, left_on='Region', right_on='State')

# # Calculate the correlation
# correlation = merged_df['Search Interest'].corr(merged_df['Overall_R_squared'])
# print("Correlation between Search Interest and Overall R-squared:", correlation)

# # Scatter plot with best fit line
# x = merged_df['Search Interest']
# y = merged_df['Overall_R_squared']

# # Best fit line
# slope, intercept, r_value, p_value, std_err = linregress(x, y)
# line = slope * x + intercept

# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color='blue')
# plt.plot(x, line, color='red', label=f'y={slope:.2f}x+{intercept:.2f}')
# plt.xlabel('Search Interest')
# plt.ylabel('Overall R-squared')
# plt.title('Correlation between Search Interest and Overall R-squared')
# plt.legend()
# plt.show()


# In[45]:


merged_avg_spei_aligned


# In[46]:


# # Merge the two datasets
# merged_df = pd.merge(rise_in_interest_df, overall_state_averages_df, left_on='Region', right_on='State')

# # Calculate the correlation
# correlation = merged_df['Rise in Interest'].corr(merged_df['Overall_R_squared'])
# print("Correlation between Rise in Search Interest and Overall R-squared:", correlation)

# # Scatter plot with best fit line
# x = merged_df['Rise in Interest']
# y = merged_df['Overall_R_squared']

# # Best fit line
# slope, intercept, r_value, p_value, std_err = linregress(x, y)
# line = slope * x + intercept

# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color='blue')
# plt.plot(x, line, color='red', label=f'y={slope:.2f}x+{intercept:.2f}')
# plt.xlabel('Rise in Search Interest')
# plt.ylabel('Overall R-squared')
# plt.title('Correlation between Rise in Search Interest and Overall R-squared')
# plt.legend()
# plt.show()


# In[47]:


# # Filter for the start and end months
# start_month_interest = usa_df[usa_df['Month'] == '2017-01'].groupby('Region')['Search Interest'].mean()
# end_month_interest = usa_df[usa_df['Month'] == '2020-12'].groupby('Region')['Search Interest'].mean()

# # Calculate the rise in search interest
# rise_in_interest = end_month_interest - start_month_interest
# rise_in_interest_df = rise_in_interest.reset_index().rename(columns={'Search Interest': 'Rise in Interest'})
# # Merge the datasets
# merged_df = pd.merge(rise_in_interest_df, overall_state_averages_df, left_on='Region', right_on='State')

# # Calculate the correlation
# correlation = merged_df['Rise in Interest'].corr(merged_df['Overall_R_squared'])
# print("Correlation between Rise in Search Interest (08/01/2017 - 12/31/2020) and Overall R-squared:", correlation)

# # Scatter plot with best fit line
# x = merged_df['Rise in Interest']
# y = merged_df['Overall_R_squared']
# slope, intercept, r_value, p_value, std_err = linregress(x, y)
# line = slope * x + intercept

# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color='blue')
# plt.plot(x, line, color='red', label=f'y={slope:.2f}x+{intercept:.2f}')
# plt.xlabel('Rise in Search Interest (08/01/2017 - 12/31/2020)')
# plt.ylabel('Overall R-squared')
# plt.title('Rise in Search Interest (for test period) and Overall R-squared')
# plt.legend()
# plt.show()


# In[48]:


# Find the model with the highest R-squared for each state
state_averages_df['Best_Model'] = state_averages_df[['Model_0ML', 'Model_1ML', 'Model_2ML', 'Model_3ML', 'Model_4ML', 'Model_5ML']].idxmax(axis=1)

# Map the model names to a categorical number for plotting
model_to_number = {'Model_0ML': '0 Month Lagged', 'Model_1ML': '1 Month Lagged', 'Model_2ML': '2 Months Lagged', 'Model_3ML': '3 Months Lagged', 'Model_4ML': '4 Months Lagged', 'Model_5ML': '5 Months Lagged'}
state_averages_df['Best_Model_Num'] = state_averages_df['Best_Model'].map(model_to_number)

# Merge with us_states for plotting
best_model_df = us_states.merge(state_averages_df, left_on='NAME', right_index=True)

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
best_model_df.plot(column='Best_Model_Num', ax=ax, cmap='Accent', categorical=True, legend=True, legend_kwds={'loc': 'upper left', 'bbox_to_anchor':(1,1)})

# Add state abbreviations as annotations on the map
for idx, row in best_model_df.iterrows():
    point = row['geometry'].representative_point().coords[:][0]
    ax.annotate(text=row['STUSPS'], xy=point, horizontalalignment='center', verticalalignment='center', fontsize=10)

ax.set_title('Best Lagged Model Per State')
ax.axis('off')

# Save the figure
plt.savefig('p5_best_model_performance_per_state.png', bbox_inches='tight')
plt.tight_layout()
plt.show()


# In[49]:


# Calculate the sum of R-squared values for each model across all states
model_performance_sums = state_averages_df[['Model_0ML', 'Model_1ML', 'Model_2ML', 'Model_3ML', 'Model_4ML', 'Model_5ML']].sum()

# Convert to DataFrame for easier plotting
model_performance_sums_df = model_performance_sums.reset_index()
model_performance_sums_df.columns = ['Model', 'Total_R_squared']

# Plotting the total R-squared values for each model
fig, ax = plt.subplots(figsize=(6, 6))
model_performance_sums_df.plot(kind='bar', x='Model', y='Total_R_squared', ax=ax, legend=False, color='r')

# Add labels and title
ax.set_xlabel('Model')
ax.set_ylabel('Total R-squared')
ax.set_title('Sum of Model R-squared')
ax.set_xticklabels(model_performance_sums_df['Model'], rotation=45)
plt.ylim(21, 22)
plt.grid()
# Save the figure
plt.savefig('p5_sum_of_model_performance_across_states.png', bbox_inches='tight')
plt.show()


# In[50]:


state_populations = {
    'Alabama': 4921532,
    'Arizona': 7421401,
    'Arkansas': 3030522,
    'California': 39237836,
    'Colorado': 5893634,
    'Connecticut': 3605944,
    'Delaware': 990837,
    'Florida': 21944577,
    'Georgia': 10830007,
    'Idaho': 1896652,
    'Illinois': 12569321,
    'Indiana': 6805985,
    'Iowa': 3190369,
    'Kansas': 2937880,
    'Kentucky': 4480713,
    'Louisiana': 4645318,
    'Maine': 1372247,
    'Maryland': 6177224,
    'Massachusetts': 6976597,
    'Michigan': 10050811,
    'Minnesota': 5707390,
    'Mississippi': 2966786,
    'Missouri': 6160281,
    'Montana': 1104271,
    'Nebraska': 1961504,
    'Nevada': 3238601,
    'New Hampshire': 1377529,
    'New Jersey': 8874520,
    'New Mexico': 2120220,
    'New York': 19336776,
    'North Carolina': 10807491,
    'North Dakota': 774948,
    'Ohio': 11714618,
    'Oklahoma': 3990443,
    'Oregon': 4325290,
    'Pennsylvania': 12804123,
    'Rhode Island': 1097379,
    'South Carolina': 5218040,
    'South Dakota': 896581,
    'Tennessee': 6975218,
    'Texas': 29730311,
    'Utah': 3363182,
    'Vermont': 643503,
    'Virginia': 8638218,
    'Washington': 7796941,
    'West Virginia': 1784787,
    'Wisconsin': 5897473,
    'Wyoming': 581075
}


# In[51]:


# Map the population data to a new column in state_averages_df
state_averages_df['Population'] = state_averages_df.index.map(state_populations)
model_columns = ['Model_0ML', 'Model_1ML', 'Model_2ML', 'Model_3ML', 'Model_4ML', 'Model_5ML']

# Calculate population-weighted performance
for model in model_columns:
    state_averages_df[f'{model}'] = state_averages_df[model] * state_averages_df['Population']
# Sum the population-weighted performances for each model
model_pop_weighted_sums = state_averages_df[[f'{model}' for model in model_columns]].sum()
# Convert series to DataFrame for easier plotting
model_pop_weighted_sums_df = model_pop_weighted_sums.reset_index()
model_pop_weighted_sums_df.columns = ['Model', 'PopWeightedSum']


# In[52]:


plt.figure(figsize=(6, 6))
plt.bar(model_pop_weighted_sums_df['Model'], model_pop_weighted_sums_df['PopWeightedSum'], color='r')
# Setting y-axis limits
plt.ylim(1.42e8, 1.43e8)
plt.xlabel('Model')
plt.ylabel('Population-Weighted \n Sum of R-squared')
plt.xticks(rotation=45)
plt.title('Population-Weighted Sum of \n Model R-squared')
plt.tight_layout()  # Adjust layout to not cut off labels
plt.grid()
plt.savefig('p5_pop_sum_of_model_performance_across_states.png', bbox_inches='tight')
plt.show()


# In[117]:


fig_paths = ['p3_avg_r_squared_per_state_overall.png', 'p3_avg_r_squared_per_state_by_region.png', 'p3_pca1_state.png', 'p3_pca_regionwise_map.png']

# Create a figure and axes for the panel
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
plt.subplots_adjust(hspace=0.2, wspace=0.2)

# Flatten the array of axes for easy iteration
axs_flat = axs.flatten()

for ax, fig_path in zip(axs_flat, fig_paths):
    img = mpimg.imread(fig_path)
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.savefig('panel_3.png', bbox_inches='tight')
plt.show()
plt.close()


# In[138]:


fig_paths = [
    'p4_search_interest_vs_r_squared_correlation.png',
    'p4_trend_search_interest_vs_r_squared_correlation.png',
    'p4_variance_search_interest_vs_r_squared_correlation.png',
    'p4_avg_spei_vs_r_squared_correlation.png',
    'p4_trend_spei_vs_r_squared_correlation.png',
    'p4_variance_spei_vs_r_squared_correlation.png'
]
# Labels for each subplot
subplot_labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']

# Creating the panel with adjusted spacing
fig, axs = plt.subplots(2, 3, figsize=(30, 18))
plt.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust spacing

# Loop over each subplot, image path, and label
for ax, fig_path, label in zip(axs.ravel(), fig_paths, subplot_labels):
    img = mpimg.imread(fig_path)
    ax.imshow(img)
    ax.axis('off')
    # Position the label in the top-left corner of the subplot
    ax.text(-0.05, 1.0, label, transform=ax.transAxes, fontsize=28, va='top', ha='left')

plt.tight_layout()
plt.savefig('panel_4.png', bbox_inches='tight')
plt.show()
plt.close()


# In[55]:


# Define the figure paths
fig_paths = [
    'p5_best_model_performance_per_state.png',
    'p5_sum_of_model_performance_across_states.png',
    'p5_pop_sum_of_model_performance_across_states.png'
]

# Set up the figure layout
fig = plt.figure(figsize=(16, 8)) 

plt.subplots_adjust(hspace=-0.9, wspace=0.2)

# Define a GridSpec with 2 rows and 2 columns
# First column is twice as wide as the second one to accommodate the larger figure 18
# This setup assumes figure 18's height is roughly equal to the combined height of figures 19 and 20
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1])

# Larger figure (Figure 18) on the left, spanning both rows
ax0 = fig.add_subplot(gs[:, 0])
img0 = mpimg.imread(fig_paths[0])
ax0.imshow(img0)
ax0.axis('off')

# Figure 19 in the top-right
ax1 = fig.add_subplot(gs[0, 1])
img1 = mpimg.imread(fig_paths[1])
ax1.imshow(img1)
ax1.axis('off')

# Figure 20 in the bottom-right, directly below figure 19
ax2 = fig.add_subplot(gs[1, 1])
img2 = mpimg.imread(fig_paths[2])
ax2.imshow(img2)
ax2.axis('off')

# Adjust layout
plt.tight_layout()
plt.savefig('panel_5.png', bbox_inches='tight')
plt.show()
plt.close()


# In[56]:


# # Reshaping the DataFrame to have model names in one column and their R-squared values in another
# model_performance_long_format = state_averages_df.melt(value_vars=['Model_0ML', 'Model_1ML', 'Model_2ML', 'Model_3ML', 'Model_4ML', 'Model_5ML'],
#                                                         var_name='Model', value_name='R_squared')
# # Creating box plots
# plt.figure(figsize=(6, 6))
# boxplot = sns.boxplot(x='Model', y='R_squared', data=model_performance_long_format, palette='Reds')

# # Rotate x-axis tick labels
# plt.xticks(rotation=90)

# # # Limiting y-axis for better readability
# # boxplot.set_ylim(0.4, 0.5)

# # Adding labels and title
# plt.xlabel('Model')
# plt.ylabel('R-squared')
# plt.title('Distribution of Model Performance (R-squared) Across All States')

# # Calculate and annotate the average R-squared value for each model
# for i, model in enumerate(model_performance_long_format['Model'].unique()):
#     # Calculate the average
#     avg_r_squared = model_performance_long_format[model_performance_long_format['Model'] == model]['R_squared'].mean()
    
#     # Annotate the plot with the average R-squared value
#     plt.text(i, avg_r_squared + 0.001, f'{avg_r_squared:.3f}', ha='center', va='bottom', color='Black', fontsize=9)

# # Adjust layout to make room for the rotated x-axis labels
# plt.tight_layout()

# # Saving the figure
# plt.savefig('model_performance_boxplots_with_averages.png', bbox_inches='tight')
# plt.show()


# In[57]:


# # Count the number of states per model
# model_counts = state_averages_df['Best_Model'].value_counts()

# # Sort the model counts by model number
# model_counts = model_counts.reindex(['Model_0ML', 'Model_1ML', 'Model_2ML', 'Model_3ML', 'Model_4ML', 'Model_5ML']).fillna(0)

# # Plotting the bar chart
# model_counts.plot(kind='bar', figsize=(10, 6), color='skyblue')
# plt.title('Number of States with Best Performance by Model')
# plt.ylabel('Number of States')
# plt.xlabel('Model')
# plt.xticks(rotation=45)
# # Save the figure
# plt.savefig('number_of_states_with_best_performance_models.png', bbox_inches='tight')
# plt.show()


# In[58]:


# # Calculate the area for each state (in square kilometers)
# us_states['Area'] = us_states.to_crs({'init': 'epsg:3857'}).area / 10**6  # Converting area from square meters to square kilometers

# # Merge the area data with the best model data
# area_model_df = us_states[['NAME', 'Area']].merge(state_averages_df[['Best_Model']], left_on='NAME', right_index=True)

# # Aggregate area by best model
# model_area = area_model_df.groupby('Best_Model')['Area'].sum()

# # Sort the model area by model number
# model_area = model_area.reindex(['Model_0ML', 'Model_1ML', 'Model_2ML', 'Model_3ML', 'Model_4ML', 'Model_5ML']).fillna(0)

# # Plotting the bar chart
# model_area.plot(kind='bar', figsize=(10, 6), color='skyblue')
# plt.title('Total Area with Best Performance by Model (in sq km)')
# plt.ylabel('Total Area (sq km)')
# plt.xlabel('Model')
# plt.xticks(rotation=45)
# # Save the figure
# plt.savefig('area_of_states_with_best_performance_models.png', bbox_inches='tight')
# plt.show()


# In[ ]:




