#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


#Load state wise google SI 
usa_df = pd.read_csv('us_search_interest.csv')
# Convert the 'Month' column to datetime 
usa_df['Month'] = pd.to_datetime(usa_df['Month'])
# Cut off the dataframe after December 2020
usa_df = usa_df[usa_df['Month'] <= '2020-12']
# List of non-CONUS state names to exclude (adjust as necessary if using abbreviations)
non_conus_states = ['Alaska', 'Hawaii', 'Puerto Rico', 'United States Virgin Islands', 'District of Columbia']
# Filter out non-CONUS states and DC
usa_df = usa_df[~usa_df['Region'].isin(non_conus_states)]
# Load the US states shapefile
us_states = gpd.read_file('cb_2018_us_state_500k.shp')

#load SPEI data
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
#Make a list of dates to iterate over and generate maps
dates = spei_df.columns[2:].to_list()


# Load data into a geopandas dataframe
gdf = gpd.GeoDataFrame(spei_df['2020-12-16'], geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])


# Load the US states shapefile
us_map = gpd.read_file('cb_2018_us_state_500k.shp')

# Filter out non-CONUS states from us_map
us_map = us_map[~us_map['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico', 'United States Virgin Islands'])]

# Correctly filter spei_df to retain 'lat', 'lon', and dates within 2004-2020
columns_to_keep = ['lat', 'lon'] + [col for col in spei_df.columns if '2003-08-01' <= col <= '2020-12-16']
spei_df_filtered = spei_df[columns_to_keep]

# Create a GeoDataFrame from the filtered SPEI DataFrame
gdf_spei = gpd.GeoDataFrame(
    spei_df_filtered,
    geometry=gpd.points_from_xy(spei_df_filtered.lon, spei_df_filtered.lat)
)
gdf_spei.crs = "EPSG:4326"  # Set the coordinate reference system to WGS84

# Assuming us_map is already filtered to exclude non-CONUS states
spei_states = sjoin(gdf_spei, us_map, how="inner", op='intersects')

# Drop non-SPEI columns to focus on SPEI values
spei_values_columns = [col for col in spei_states.columns if '2004-01-16' <= col <= '2020-12-16']


# In[3]:


# List all column names to check their format
print(spei_states.columns)


# In[4]:


# Directly selecting columns based on a startswith check for year ranges
years = [str(year) for year in range(2003, 2021)]
spei_relevant_columns = ['STUSPS'] + [col for col in spei_states.columns if any(col.startswith(year) for year in years)]
spei_relevant = spei_states[spei_relevant_columns]


# In[5]:


spei_relevant


# In[6]:


spei_long = spei_relevant.melt(id_vars=['STUSPS'], var_name='Date', value_name='SPEI')
spei_long
# Ensure 'Date' is correctly formatted as 'YYYY-MM-DD' in the column names
spei_long['Date'] = spei_long['Date'].apply(lambda x: pd.to_datetime(x, errors='coerce'))

# Convert 'Date' to 'YYYY-MM' format
spei_long['Month'] = spei_long['Date'].dt.to_period('M').astype(str)

# Now, aggregate by State and Month to calculate average SPEI
average_spei = spei_long.groupby(['STUSPS', 'Month']).SPEI.mean().reset_index()

# Rename columns for merging
average_spei.rename(columns={'STUSPS': 'State'}, inplace=True)


# In[7]:


average_spei


# In[8]:


# Convert 'Month' to YYYY-MM format
usa_df['Month'] = pd.to_datetime(usa_df['Month']).dt.to_period('M').astype(str)

# Remove 'US-' prefix from 'Region ID' to get state abbreviations
usa_df['State'] = usa_df['Region ID'].str.replace('US-', '')

# Select relevant columns
usa_df = usa_df[['Month', 'State', 'Search Interest']]


# In[9]:


usa_df


# In[15]:


# Merge on 'State' and 'Month'
merged_df = pd.merge(usa_df, average_spei, how='inner', on=['State', 'Month'])


# In[17]:


merged_df


# In[12]:


# for lag in range(0, 6):  # 0 to 5 months lag
#     merged_df[f'SPEI_lag_{lag}'] = merged_df.groupby('State')['SPEI'].shift(lag)

# # Drop rows with NaN values that result from lagging (especially for the first 5 months of data per state)
# merged_df.dropna(inplace=True)


# In[18]:


# Selecting SPEI, its lagged features, and Search Interest for PCA
features = ['SPEI', 'Search Interest']
x = merged_df[features].values

# It's important to standardize the features before applying PCA
x = StandardScaler().fit_transform(x)

# Apply PCA
pca = PCA(n_components=2)  # Adjust n_components 
principalComponents = pca.fit_transform(x)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

# Print the explained variance ratio of each component
print("Explained Variance Ratio:", pca.explained_variance_ratio_)


# In[20]:


pca_df


# In[21]:


# # Reset the index of 'merged_df' if it's not already a simple integer index to ensure a clean merge
# merged_df_reset = merged_df.reset_index(drop=True)

# Concatenate the PCA scores with the original DataFrame
final_df = pd.concat([merged_df, pca_df], axis=1)

final_df


# In[ ]:


#Spatial Analysis


# In[22]:


# Convert 'Month' to datetime for easier plotting
final_df['Month'] = pd.to_datetime(final_df['Month'])

# Group by Month and calculate mean PC1 score for each month
monthly_avg_pca = final_df.groupby('Month')['PC1'].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(final_df['Month'], final_df['PC1'], marker='o', linestyle='-', color='blue')
plt.title('Average PCA Component 1 Score Over Time')
plt.xlabel('Month')
plt.ylabel('Average PC1 Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[ ]:


# # Placeholder for explained variance ratios and correlations for visualization
# explained_variances = []
# correlations = []

# for lag in range(6):  # For each lag (0 to 5 months)
#     # Select features for the current lag and standardize
#     features = [f'SPEI_lag_{lag}']
#     x = StandardScaler().fit_transform(merged_df[features].values)
    
#     # Perform PCA
#     pca = PCA(n_components=1)  # Using 1 component for simplicity in demonstration
#     principalComponents = pca.fit_transform(x)
    
#     # Capture explained variance
#     explained_variance = pca.explained_variance_ratio_[0]
#     explained_variances.append(explained_variance)
    
#     # Calculate correlation with search interest
#     search_interest = merged_df['Search Interest'].values.reshape(-1, 1)
#     correlation = np.corrcoef(principalComponents.flatten(), search_interest.flatten())[0, 1]
#     correlations.append(correlation)

# # Visualize the results
# lags = range(6)
# plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# plt.bar(lags, explained_variances, color='skyblue')
# plt.xlabel('SPEI Lag (months)')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained Variance by SPEI Lag')

# plt.subplot(1, 2, 2)
# plt.plot(lags, correlations, marker='o', linestyle='-', color='orange')
# plt.xlabel('SPEI Lag (months)')
# plt.ylabel('Correlation with Search Interest')
# plt.title('Correlation by SPEI Lag')

# plt.tight_layout()
# plt.show()


# In[ ]:


# correlations = []
# for lag in range(6):  # For each lag (0 to 5 months)
#     # Calculate correlation between each lagged SPEI and search interest
#     correlation = merged_df[['SPEI_lag_' + str(lag), 'Search Interest']].corr().iloc[0, 1]
#     correlations.append(correlation)

# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(range(6), correlations, marker='o', linestyle='-', color='blue')
# plt.xlabel('SPEI Lag (months)')
# plt.ylabel('Correlation with Search Interest')
# plt.title('Correlation of Lagged SPEI with Search Interest')
# plt.xticks(range(6))
# plt.grid(True)
# plt.show()


# In[19]:


gdf_states = gpd.read_file('cb_2018_us_state_500k.shp')


# In[ ]:


# # Assuming merged_df has 'State', 'Month', and SPEI lags as features
# # features = [col for col in merged_df.columns if 'SPEI_lag' in col]
# features = ['SPEI', 'Search Interest']
# X = merged_df[features].values


# # Standardizing the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Perform PCA
# pca = PCA(n_components=2)  # Starting with 2 components as in Kim's paper
# principalComponents = pca.fit_transform(X_scaled)
# explained_variance = pca.explained_variance_ratio_

# # Creating a DataFrame for the PCA results
# pca_results = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

# merged_df['Month'] = pd.to_datetime(merged_df['Month'])



# # Add back the 'State' and 'Month' for visualization
# pca_results['State'] = merged_df['State']
# pca_results['Month'] = merged_df['Month']

# # Visualization: Explained Variance
# plt.figure(figsize=(8, 4))
# plt.bar(['PC1', 'PC2'], explained_variance*100)
# plt.ylabel('Percentage of Variance Explained')
# plt.title('PCA Explained Variance')
# plt.savefig('pca_overall.png', bbox_inches='tight')


# In[23]:


pca_results = final_df

monthly_avg_pca = pca_results.groupby('Month').mean().reset_index()

# Plotting both PCA Component 1 and 2 scores over time side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot for PCA Component 1 (PC1)
axes[0].plot(monthly_avg_pca['Month'], monthly_avg_pca['PC1'], marker='o', linestyle='-', color='blue')
axes[0].set_title('Average PCA Component 1 Score Over Time')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Average PC1 Score')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid()

# Plot for PCA Component 2 (PC2)
axes[1].plot(monthly_avg_pca['Month'], monthly_avg_pca['PC2'], marker='o', linestyle='-', color='green')
axes[1].set_title('Average PCA Component 2 Score Over Time')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Average PC2 Score')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid()

plt.tight_layout()# Adjusts subplot params so that subplots fit into the figure area.
plt.savefig('pca_timeseries.png', bbox_inches='tight')

plt.show()


# In[24]:


# Assuming pca_results contains 'State', 'PC1', and 'PC2'
pca_aggregated = pca_results.groupby('State').mean().reset_index()

# Ensure the state identifier matches between pca_aggregated and gdf_states
gdf_states['State'] = gdf_states['STUSPS']  # Update 'state_code' to actual identifier column

# Merge on 'State'
gdf_pca = gdf_states.merge(pca_aggregated, on='State')


# In[36]:


fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plotting the GeoDataFrame with PCA Component 1 scores
pca_plot = gdf_pca.plot(column='PC1', ax=ax, legend=True, cmap = 'RdBu',
                        legend_kwds={'orientation': "horizontal", 'shrink': 0.5})

# After plotting, find the colorbar in the figure
cbar = fig.axes[-1]  # The colorbar should be the last axes object in the figure

# Set the font size of the colorbar's tick labels
cbar.tick_params(labelsize = 20)  

# Remove axes
ax.set_axis_off()

# Add state labels
for idx, row in gdf_pca.iterrows():
    ax.annotate(s=row['STUSPS'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                horizontalalignment='center', fontsize=10)

# Adjust title
plt.title('PCA Component 1 Scores by State', fontsize=30)

# # Manually create a colorbar
# sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=gdf_pca['PC1'].min(), vmax=gdf_pca['PC1'].max()))
# sm._A = []  # Empty array for the data range
# cbar = fig.colorbar(sm, orientation='horizontal', fraction=0.046, pad=0.04)
# cbar.set_label('PCA Component 1 Score')
plt.savefig('p3_pca1_state.png', bbox_inches='tight')
plt.show()


# In[26]:


# Adjust the default font size for all plot elements
plt.rcParams.update({'font.size': 18})

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Plot for PCA Component 1 (PC1)
gdf_pca.plot(column='PC1', ax=axes[0], legend=True, cmap = 'RdYlGn',
             legend_kwds={'label': "PCA Component 1 Score",
                          'orientation': "horizontal"})
axes[0].set_title('PCA Component 1 Scores by State')

# Plot for PCA Component 2 (PC2)
gdf_pca.plot(column='PC2', ax=axes[1], legend=True, cmap = 'RdYlGn',
             legend_kwds={'label': "PCA Component 2 Score", 
                          'orientation': "horizontal"})
axes[1].set_title('PCA Component 2 Scores by State')

plt.tight_layout()
plt.savefig('pca_statewise.png', bbox_inches='tight')

plt.show()


# In[27]:


# spei_df


# In[30]:


# Define regions
regions = {
    'Western': ['Arizona', 'California', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Oregon', 'Utah', 'Washington', 'Wyoming'],
    'Central': ['Alabama', 'Arkansas', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Nebraska', 'North Dakota', 'Ohio', 'Oklahoma', 'South Dakota', 'Tennessee', 'Texas', 'Wisconsin'],
    'Northeastern': ['Connecticut', 'Delaware', 'Maine', 'Maryland', 'Massachusetts', 'New Hampshire', 'New Jersey', 'New York', 'Pennsylvania', 'Rhode Island', 'Vermont', 'Virginia', 'West Virginia'],
    'Southeastern': ['Florida', 'Georgia', 'North Carolina', 'South Carolina']
}


# State abbreviations 
state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT',
    'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN',
    'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Create a dictionary that maps state abbreviations to regions
regions_abbreviations = {state_abbreviations[state]: region for region, states in regions.items() for state in states}

# Assuming pca_aggregated contains 'State', 'PC1', and 'PC2' with 'State' being state abbreviations
pca_aggregated['Region'] = pca_aggregated['State'].map(regions_abbreviations)

# Aggregate PCA scores by region
region_pca_scores = pca_aggregated.groupby('Region').mean().reset_index()

# Assuming gdf_states is a GeoDataFrame with geometries for each state and 'STUSPS' contains state abbreviations
gdf_states['Region'] = gdf_states['STUSPS'].map(regions_abbreviations)

# Dissolve gdf_states by 'Region' to get regional geometries
gdf_regions = gdf_states.dissolve(by='Region', aggfunc='mean')

# Merge the regional PCA scores with the gdf_regions
gdf_pca_regions = gdf_regions.merge(region_pca_scores, on='Region')

# Visualization
fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(20, 10))

# Increase the base font size
plt.rcParams.update({'font.size': 18})

# Map: Regional PCA Component 1 Scores
gdf_pca_regions.plot(column='PC1', ax=ax_map, legend=True, cmap='RdBu',
                     legend_kwds={'label': "PCA Component 1 Score", 'orientation': "horizontal"})
ax_map.set_title('Regional PCA Component 1 Scores', fontsize=20)
ax_map.set_axis_off()

# Bar Chart: Regional PCA Component 1 Scores
ax_bar.bar(region_pca_scores['Region'], region_pca_scores['PC1'], color='skyblue')
ax_bar.set_title('PCA Component 1 Scores by Region', fontsize=20)
ax_bar.set_xlabel('Region', fontsize=18)
ax_bar.set_ylabel('Average PC1 Score', fontsize=18)
ax_bar.tick_params(axis='x', rotation=45, labelsize=16)
ax_bar.tick_params(axis='y', labelsize=16)
ax_bar.grid()

# Adjust the color bar size
# cbar = plt.cm.ScalarMappable(cmap='coolwarm')
# cbar.set_array([])
# cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])  # Position for the color bar
# fig.colorbar(cbar, cax=cbar_ax)
# cbar_ax.set_ylabel('PCA Component 1 Score', fontsize=18)

plt.tight_layout()
plt.subplots_adjust(right=0.9)  # Adjust right to make room for the color bar
plt.savefig('p3_pca_regionwise.png', bbox_inches='tight')
plt.show()


# In[37]:


fig, ax = plt.subplots(1, figsize=(15, 10))

# Plotting the map with regional PCA Component 1 scores
gdf_pca_regions.plot(column='PC1', ax=ax, legend=True, cmap='RdBu',
                     legend_kwds={'orientation': "horizontal",
                                  'shrink': 0.5})  # Adjust the legend size
# # Overlay state boundaries
# gdf_states.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
# Remove axes
ax.set_axis_off()

# Add state labels for better readability
for idx, row in gdf_states.iterrows():
    ax.annotate(s=row['STUSPS'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                horizontalalignment='center', fontsize=10)

# Adjust title
plt.title('Regional PCA Component 1 Scores', fontsize=30)

plt.savefig('p3_pca_regionwise_map.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:


# Remove rows where either 'State' or 'Month' is NaN
pca_results_cleaned = pca_results.dropna(subset=['State', 'Month']).reset_index(drop=True)
pca_results_cleaned


# In[ ]:


# Make sure both DataFrames are sorted the same way and have the same format for 'Month' and 'State'
merged_df['Month'] = pd.to_datetime(merged_df['Month'])
pca_results_cleaned['Month'] = pd.to_datetime(pca_results_cleaned['Month'])

# Assuming 'State' in both DataFrames are already matching formats
# Merge PCA scores from pca_results_cleaned into merged_df based on 'State' and 'Month'
merged_with_pca = pd.merge(merged_df, pca_results_cleaned, on=['State', 'Month'], how='left')


# In[ ]:


merged_with_pca_cleaned = merged_with_pca.dropna()


# In[ ]:


# Initialize a DataFrame to store correlation results
correlations = []

# Calculate correlations for each state
for state in merged_with_pca_cleaned['State'].unique():
    state_data = merged_with_pca_cleaned[merged_with_pca_cleaned['State'] == state]
    
    # Calculate and store the correlation of PC1 with SPEI and Search Interest
    pc1_spei_corr = state_data['PC1'].corr(state_data['SPEI'])
    pc1_search_corr = state_data['PC1'].corr(state_data['Search Interest'])
    
    # Calculate and store the correlation of PC2 with SPEI and Search Interest
    pc2_spei_corr = state_data['PC2'].corr(state_data['SPEI'])
    pc2_search_corr = state_data['PC2'].corr(state_data['Search Interest'])
    
    # Append the results
    correlations.append({
        'State': state,
        'PC1_SPEI_Corr': pc1_spei_corr,
        'PC1_Search_Corr': pc1_search_corr,
        'PC2_SPEI_Corr': pc2_spei_corr,
        'PC2_Search_Corr': pc2_search_corr
    })

# Convert the list of correlations to a DataFrame
correlation_df = pd.DataFrame(correlations)

# Display the first few rows to verify
print(correlation_df.head())


# In[ ]:


correlation_df


# In[ ]:


# Merge the correlation data with the GeoDataFrame
gdf_correlation = gdf_states.merge(correlation_df, on='State', how='left')

# Determine common scale for all plots based on the correlation range
vmin = min(gdf_correlation[['PC1_SPEI_Corr', 'PC1_Search_Corr', 'PC2_SPEI_Corr', 'PC2_Search_Corr']].min())
vmax = max(gdf_correlation[['PC1_SPEI_Corr', 'PC1_Search_Corr', 'PC2_SPEI_Corr', 'PC2_Search_Corr']].max())

# Ensure 0 is centered and represented as white by adjusting vmin and vmax to be equidistant from 0
vmax = max(abs(vmin), abs(vmax))
vmin = -vmax

# Set up the matplotlib figure and axes for the 2x2 subplot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15), sharex=True, sharey=True)

# Adjust overall plot spacing
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Define colormap that centers 0 as white
cmap = plt.cm.coolwarm
cmap.set_bad('white')  # Set NaN values to white if necessary
cmap.set_under('blue')  # Set values under vmin to blue if necessary
cmap.set_over('red')    # Set values over vmax to red if necessary

# Mapping configurations for the first row (Search Interest)
mapping_configs_first_row = [
    ('PC1_Search_Corr', 'PC1 and Search Interest Correlation', axes[0, 0]),
    ('PC2_Search_Corr', 'PC2 and Search Interest Correlation', axes[0, 1])
]

# Mapping configurations for the second row (SPEI)
mapping_configs_second_row = [
    ('PC1_SPEI_Corr', 'PC1 and SPEI Correlation', axes[1, 0]),
    ('PC2_SPEI_Corr', 'PC2 and SPEI Correlation', axes[1, 1])
]

# Plot first row
for column, title, ax in mapping_configs_first_row:
    gdf_correlation.plot(column=column, ax=ax, legend=False, cmap=cmap,
                         vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=15)
    ax.axis('off')

# Plot second row
for column, title, ax in mapping_configs_second_row:
    gdf_correlation.plot(column=column, ax=ax, legend=False, cmap=cmap,
                         vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=15)
    ax.axis('off')

# Add a common colorbar
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
             ax=axes.ravel().tolist(), orientation='horizontal', pad=0.02, aspect=40, shrink=0.75, label='Correlation')

plt.savefig('pca_correlation_spei_si.png', bbox_inches='tight')
plt.show()


# In[ ]:


# Add a 'Region' column to the correlation DataFrame
correlation_df['Region'] = correlation_df['State'].map(regions_abbreviations)

# Now group by region and calculate the average correlation for PC1 with Search Interest
region_correlation = correlation_df.groupby('Region')['PC1_Search_Corr'].mean().reset_index()

# If you want to see the results
print(region_correlation)
plt.bar(region_correlation['Region'], region_correlation['PC1_Search_Corr'])

