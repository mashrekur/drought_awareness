#!/usr/bin/env python
# coding: utf-8

# In[108]:


import geopandas as gpd
import requests
from zipfile import ZipFile
from io import BytesIO
import os
import xarray as xr
from shapely.geometry import Point
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.collections import PatchCollection
import pandas as pd
from shapely.geometry import Polygon
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.stats import pearsonr


# In[2]:


# Download the US states shapefile from the US Census Bureau
url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip"
response = requests.get(url)
zipfile = ZipFile(BytesIO(response.content))


# In[3]:


# Extract the contents of the zipfile
zipfile.extractall()

# Read the shapefile
states_gdf = gpd.read_file("cb_2020_us_state_20m.shp")


# In[4]:


# Filter out non-CONUS states (Hawaii, Alaska, and island territories)
conus_states_gdf = states_gdf[~states_gdf["STUSPS"].isin(["AK", "HI", "AS", "GU", "MP", "PR", "VI"])]


# In[5]:


# Create a list of state geometries
state_geometries = list(conus_states_gdf["geometry"])


# In[ ]:


# Open the NETCDF file using xarray
ds = xr.open_dataset('data/spei01.nc')

# Convert the xarray dataset to a pandas dataframe
df = ds.to_dataframe()

df = df.unstack()


# In[ ]:


spei_df = df.dropna()
spei_df = spei_df['spei']
spei_df = spei_df.reset_index()
spei_df = spei_df.apply(lambda x: x.ffill())


# In[ ]:


# Convert all column names to str and limit the size to 10 characters since the hh:mm:ss are all the same
spei_df.columns = spei_df.columns.astype(str)
spei_df.columns = spei_df.columns.str.slice(stop=10)


# In[ ]:


spei_df.to_csv('spei_df.csv', index=False)


# In[6]:


spei_df = pd.read_csv('spei_df.csv')


# In[89]:


spei_df


# In[7]:


#Make a list of dates to iterate over and generate maps
dates = spei_df.columns[1238:].to_list()
dates_1ml = spei_df.columns[1237:-1].to_list()
dates_2ml = spei_df.columns[1236:-2].to_list()
dates_3ml = spei_df.columns[1235:-3].to_list()
dates_4ml = spei_df.columns[1234:-4].to_list()
dates_5ml = spei_df.columns[1233:-5].to_list()
dates_6ml = spei_df.columns[1232:-6].to_list()


# In[9]:


# for date in dates:
#     # Load data into a geopandas dataframe
#     gdf = gpd.GeoDataFrame(spei_df, geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])

#     # Set the CRS of gdf to match us_map
#     gdf.set_crs(conus_states_gdf.crs, inplace=True)

#     # Normalize the data
#     norm = Normalize(vmin=gdf[date].min(), vmax=gdf[date].max())

#     # cmap
#     cmap = plt.get_cmap('gray')
#     cmap = ListedColormap(cmap(np.linspace(0.0, 1.0, 256)))
#     cmap.set_bad(color='white')

#     for index, row in conus_states_gdf.iterrows():
#         state_name = row['NAME']
#         state_geometry = row['geometry']

#         # Clip the points to the current state
#         state_gdf = gpd.clip(gdf, state_geometry)

#         # Create a patch collection
#         square_size = 0.5
#         patches = [Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size) for x, y in zip(state_gdf.geometry.x, state_gdf.geometry.y)]
#         pc = PatchCollection(patches, facecolor=cmap(norm(state_gdf[date])), alpha=0.8, edgecolor='gray')

#         # Plot the state
#         ax = gpd.GeoSeries(state_geometry).plot(figsize=(15, 10), color='white', edgecolor='black')

#         # Add the patch collection to the map
#         ax.add_collection(pc)

#         # Plot the state border on top
#         gpd.GeoSeries(state_geometry.boundary).plot(ax=ax, linewidth=1, edgecolor='black')

#         # Create a directory for the state if it doesn't exist
#         state_dir = f'state_maps_spei_0ml/{state_name}'
#         if not os.path.exists(state_dir):
#             os.makedirs(state_dir)

#         # Save the map as a file
#         ax.get_figure().savefig(f'{state_dir}/{state_name}_{date}_map.png')

#         # Close the figure
#         ax.get_figure().clf()


# In[130]:


#Load state wise google SI 
usa_df = pd.read_csv('us_search_interest.csv')
usa_df


# In[131]:


usa_df['Search Interest'].isna().sum()


# In[132]:


# # Define the start and end dates
# start_date = '2004-01'
# end_date = '2022-12'

# # Define the date range for the loop
# date_range = pd.date_range(start=start_date, end=end_date, freq='MS')


# In[133]:


# Convert the 'Month' column to datetime
usa_df['Month'] = pd.to_datetime(usa_df['Month'])
usa_df


# In[14]:


# Create a plot figure
plt.figure(figsize=(12, 8))

# Find regions that have 'Search Interest' over 50 at least once
regions_to_plot = usa_df[usa_df['Search Interest'] > 95]['Region'].unique()

# Iterate over each unique region to create separate plots for each
for region in regions_to_plot:
    region_df = usa_df[usa_df['Region'] == region]
    plt.plot(region_df['Month'], region_df['Search Interest'], label=region)

# Add a legend
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

# Add title and labels
plt.title('Search Interest Over Time for Each Region')
plt.xlabel('Month')
plt.ylabel('Search Interest')

# Show the plot
plt.show()


# In[15]:


# Find the top 10 states with the highest 'Search Interest'
top_states = usa_df.groupby('Region')['Search Interest'].max().nlargest(10).index

# Create a plot figure
plt.figure(figsize=(30, 8))

# Iterate over each state to create separate plots for each
for state in top_states:
    state_df = usa_df[usa_df['Region'] == state]
    plt.plot(state_df['Month'], state_df['Search Interest'], label=state)

# Add a legend
plt.legend()

# Add title and labels
plt.title('Search Interest Over Time for Top 10 States')
plt.xlabel('Month')
plt.ylabel('Search Interest')

# Show the plot
plt.show()


# In[16]:


# Calculate the average 'Search Interest' for each state
average_interest = usa_df.groupby('Region')['Search Interest'].mean()

# Get the top 10 states
top_10_states = average_interest.nlargest(10)

# Get the bottom 10 states
bottom_10_states = average_interest.nsmallest(10)

# Find the maximum average 'Search Interest' across all states
max_avg_interest = average_interest.max()

# Create the subplot figure
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

# Plot the top 10 states
top_10_states.sort_values().plot(kind='barh', ax=axs[0], color='b')
axs[0].set_title('Top 10 States by Average Search Interest')
axs[0].set_xlabel('Average Search Interest')
axs[0].grid(True)

# Plot the bottom 10 states
bottom_10_states.sort_values().plot(kind='barh', ax=axs[1], color='r')
axs[1].set_title('Bottom 10 States by Average Search Interest')
axs[1].set_xlabel('Average Search Interest')
axs[1].grid(True)

# Set the same x-axis limits for both subplots
axs[0].set_xlim([0, max_avg_interest])
axs[1].set_xlim([0, max_avg_interest])

plt.tight_layout()

# Save the figure as a PNG image
plt.savefig("average_search_interest.png", dpi=300)

# Save the figure as a PDF
plt.savefig("average_search_interest.pdf")

# Show the plot
plt.show()


# In[17]:


# Filter the dataframe to keep only the data up to the end of 2020
usa_df = usa_df[usa_df['Month'] <= '2020-12']

# Calculate the average 'Search Interest' for each state
average_interest = usa_df.groupby('Region')['Search Interest'].mean()

# Get the top 10 states
top_10_states = average_interest.nlargest(10)

# Get the bottom 10 states
bottom_10_states = average_interest.nsmallest(10)

# Calculate the change in yearly 'Search Interest' for each state
usa_df['Year'] = usa_df['Month'].dt.year
yearly_interest = usa_df.groupby(['Year', 'Region'])['Search Interest'].mean().unstack()
rise_in_interest = yearly_interest.loc[2020] - yearly_interest.loc[2004]

# Get the top 10 states with the highest rise
top_10_rise_states = rise_in_interest.nlargest(10)

# Create the subplot figure
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))

# Define xticks range
xticks_range = range(0, 101, 10)

# Adjust font sizes for ticks, labels, and titles
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

# Plot the top 10 states
top_10_states.sort_values().plot(kind='barh', ax=axs[0], color='b', xticks=xticks_range)
axs[0].set_title('Top 10 States by Average Search Interest', fontsize=24)
axs[0].set_xlabel('Average Search Interest', fontsize=20)
axs[0].grid(True)
axs[0].set_xlim([0, 100])

# Plot the bottom 10 states
bottom_10_states.sort_values().plot(kind='barh', ax=axs[1], color='r', xticks=xticks_range)
axs[1].set_title('Bottom 10 States by Average Search Interest', fontsize=24)
axs[1].set_xlabel('Average Search Interest', fontsize=20)
axs[1].grid(True)
axs[1].set_xlim([0, 100])

# Plot the top 10 states with the highest rise
top_10_rise_states.sort_values().plot(kind='barh', ax=axs[2], color='g', xticks=xticks_range)
axs[2].set_title('Top 10 States by Rise in Yearly Search Interest', fontsize=24)
axs[2].set_xlabel('Rise in Yearly Search Interest', fontsize=20)
axs[2].grid(True)
axs[2].set_xlim([0, 100])

plt.tight_layout()

# Save the figure as a PNG image
plt.savefig("search_interest_analysis.png", dpi=300)

# Save the figure as a PDF
plt.savefig("search_interest_analysis.pdf")

# Show the plot
plt.show()


# In[90]:


spei_df_cut = spei_df.iloc[:, list(range(2)) + list(range(-205, 0))]
spei_df_cut


# In[91]:


# Convert lon/lat to a shapely Point object
spei_df_cut["geometry"] = spei_df_cut.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)

# Convert DataFrame to GeoDataFrame
spei_gdf = gpd.GeoDataFrame(spei_df_cut, geometry='geometry')

# Set the coordinate reference system (CRS) for spei_gdf to match that of conus_states_gdf
spei_gdf.crs = conus_states_gdf.crs

# Perform the spatial join
joined_gdf = gpd.sjoin(spei_gdf, conus_states_gdf, how='inner', op='within')

# Drop unexpected column if exists
if 'index_right' in joined_gdf.columns:
    joined_gdf = joined_gdf.drop(columns=['index_right'])

# Get datetime columns only (columns with year-month-day format)
datetime_columns = [col for col in joined_gdf.columns if "-" in str(col)]

# Melt the DataFrame to have one row per date
melted_df = pd.melt(joined_gdf, id_vars=["lon", "lat", "geometry", "STUSPS"], value_vars=datetime_columns, var_name="date", value_name="SPEI")
melted_df["date"] = pd.to_datetime(melted_df["date"])

# Calculate yearly SPEI
melted_df["year"] = melted_df["date"].dt.year
yearly_spei_df = melted_df.groupby(["year", "STUSPS"])["SPEI"].mean().reset_index()

# Pivot DataFrame to have years as rows and states as columns
pivot_df = yearly_spei_df.pivot(index="year", columns="STUSPS", values="SPEI")


# In[92]:


pivot_df


# In[22]:


# Extract year and month from the date column
melted_df['year'] = melted_df['date'].dt.year
melted_df['month'] = melted_df['date'].dt.month

# Group by year and month and calculate the mean
monthly_spei_df = melted_df.groupby(['year', 'month', 'STUSPS'])['SPEI'].mean().reset_index()

# Pivot DataFrame to have years and months as rows and states as columns
pivot_df_monthly = monthly_spei_df.pivot_table(index=['year', 'month'], columns='STUSPS', values='SPEI')


# In[23]:


pivot_df_monthly


# In[24]:


# Initialize a dictionary to hold state codes and their trend coefficients
trends = {}

# Loop through each state (column in the pivot table)
for state in pivot_df_monthly.columns:
    # Prepare the input data
    X = np.arange(len(pivot_df_monthly)).reshape(-1, 1)  # Time variable
    Y = pivot_df_monthly[state].values  # SPEI values
    X = sm.add_constant(X)  # Add a constant to the model

    # Handle any missing data
    mask = ~np.isnan(Y)
    X = X[mask]
    Y = Y[mask]

    # Run the OLS model
    model = sm.OLS(Y, X)
    results = model.fit()

    # Store the slope (trend) in the dictionary
    trends[state] = results.params[1]

# Convert the dictionary to a DataFrame and sort it by the trend
trends_df = pd.DataFrame.from_dict(trends, orient='index', columns=['Trend'])
trends_df = trends_df.sort_values('Trend')

# Plot the top 20 states with strongest decreasing trend
top_20_decreasing = trends_df.head(20)
top_20_decreasing.plot(kind='barh', figsize=(10, 8), legend=False)
plt.xlabel('Trend')
plt.title('Top 20 states with strongest decreasing monthly SPEI trend')
plt.show()


# In[25]:


# Initialize a dictionary to hold state codes and their trend coefficients
trends = {}

# Loop through each state (column in the pivot table)
for state in pivot_df.columns:
    # Prepare the input data
    X = np.arange(len(pivot_df)).reshape(-1, 1)  # Time variable
    Y = pivot_df[state].values  # SPEI values
    X = sm.add_constant(X)  # Add a constant to the model

    # Handle any missing data
    mask = ~np.isnan(Y)
    X = X[mask]
    Y = Y[mask]

    # Fit the model and calculate the trend
    model = sm.OLS(Y, X)
    results = model.fit()
    trends[state] = results.params[1]  # Store the coefficient of the time variable

# Convert the dictionary to a DataFrame for easy handling
trends_df = pd.DataFrame.from_dict(trends, orient='index', columns=['trend'])

# Sort the DataFrame by the trend coefficient in ascending order
trends_df = trends_df.sort_values(by='trend')

# Let's plot the states with the strongest decreasing SPEI trend (top 20)
top20_dec_trends = trends_df.head(20)
# top20_dec_trends.plot(kind='barh', figsize=(10, 6), legend=False)
# plt.xlabel('SPEI trend (slope of regression line fitted to yearly data)')
# plt.title('Top 20 states with strongest decreasing SPEI trend')
# plt.grid()
# plt.show()
# Get the top 20 trends
top20_dec_trends = trends_df.sort_values('trend').head(20)

# Plot using Seaborn
plt.figure(figsize=(10,8))
sns.barplot(x='trend', y=top20_dec_trends.index, data=top20_dec_trends, palette="coolwarm")
plt.xlabel('SPEI trend (slope of regression line fitted to yearly data)')
plt.title('Top 20 states with strongest decreasing SPEI trend')
plt.grid(True)
plt.show()


# In[26]:


# Define a list of non-CONUS states
non_conus_states = ['Alaska', 'Hawaii']

# Filter the dataframe to remove non-CONUS states
usa_df_conus = usa_df[~usa_df['Region'].isin(non_conus_states)]

# Group the dataframe by Region and iterate over the groups
trends = {}
for region, group in usa_df_conus.groupby('Region'):
    # Extract the year and Search Interest
    year = group['Month'].dt.year.to_numpy().reshape(-1, 1)
    search_interest = group['Search Interest'].to_numpy().reshape(-1, 1)

    # Calculate linear regression
    model = LinearRegression()
    model.fit(year, search_interest)
    trend = model.coef_[0][0]

    # Store the trend
    trends[region] = trend

# Sort states by the trend and take the top 20
top_20_states_search_interest = sorted(trends.items(), key=lambda x: x[1], reverse=True)[:20]

# Convert to DataFrame for easier plotting
top_20_states_search_interest_df = pd.DataFrame(top_20_states_search_interest, columns=['Region', 'Trend'])

# Plot
plt.figure(figsize=(10,8))
sns.barplot(data=top_20_states_search_interest_df, y='Region', x='Trend', palette='viridis')
plt.title('Top 20 CONUS states with highest trend in Search Interest')
plt.xlabel('Trend')
plt.ylabel('State')
plt.grid()
plt.show()


# In[27]:


# Group the dataframe by Region and Region ID and iterate over the groups
trends = {}
for (region, region_id), group in usa_df.groupby(['Region', 'Region ID']):
    # Extract the year and Search Interest
    year = group['Month'].dt.year.to_numpy().reshape(-1, 1)
    search_interest = group['Search Interest'].to_numpy().reshape(-1, 1)

    # Calculate linear regression
    model = LinearRegression()
    model.fit(year, search_interest)
    trend = model.coef_[0][0]

    # Store the trend with Region ID
    trends[region_id] = trend

# Sort states by the trend and take the top 20
top_20_states_search_interest = sorted(trends.items(), key=lambda x: x[1], reverse=True)[:20]

# Convert to DataFrame for easier plotting
top_20_states_search_interest_df = pd.DataFrame(top_20_states_search_interest, columns=['Region ID', 'Trend'])

# Strip 'US-' from 'Region ID' to get abbreviated state names
top_20_states_search_interest_df['State'] = top_20_states_search_interest_df['Region ID'].str.slice(start=3)


# In[28]:


# Find common states
common_states = set(top_20_states_search_interest_df['State']).intersection(set(top20_dec_trends.index))

# Data for the third plot
common_states_data = [{'State': state, 'Data': 'SPEI'} for state in common_states] + [{'State': state, 'Data': 'Search Interest'} for state in common_states]
common_states_df = pd.DataFrame(common_states_data)

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

# First plot
sns.barplot(ax=axes[0], data=top20_dec_trends.reset_index(), y='index', x='trend', palette='coolwarm')
axes[0].set_title('Top 20 states with strongest decreasing SPEI trend (2004-2020)')
axes[0].set_xlabel('SPEI trend (slope of regression line fitted to yearly data)')
axes[0].set_ylabel('State')
axes[0].grid()

# Second plot
sns.barplot(ax=axes[1], data=top_20_states_search_interest_df, y='State', x='Trend', palette='coolwarm')
axes[1].set_title('Top 20 states with strongest increasing trend in Search Interest (2004-2020)')
axes[1].set_xlabel('SI Trend (slope of regression line fitted to yearly data)')
axes[1].set_ylabel('State')
axes[1].grid()

# # Third plot
# sns.countplot(ax=axes[2], data=common_states_df, y='State', hue='Data', palette='viridis')
# axes[2].set_title('Common states in top 20 SPEI trend and Search Interest trend')
# axes[2].set_xlabel('Count')
# axes[2].set_ylabel('State')
# axes[2].grid()

plt.tight_layout()

# Save the figure as a PNG image
plt.savefig("spei_si_trend.png", dpi=300)

# Save the figure as a PDF
plt.savefig("spei_si_trend.pdf")

plt.show()


# In[135]:


# List of non-CONUS states
non_conus_states = ['AK', 'HI', 'PR', 'VI', 'GU', 'MP', 'AS']

# Filter out the non-CONUS states from the Google Search Interest data
usa_df = usa_df[~usa_df['Region ID'].isin(non_conus_states)]

# Define a list of non-CONUS states
non_conus_states = ['Alaska', 'Hawaii']

# Filter the dataframe to remove non-CONUS states
usa_df = usa_df[~usa_df['Region'].isin(non_conus_states)]


# Define the time periods
time_periods = [(2004, 2010), (2011, 2015), (2016, 2020)]

# Loop through each time period
for start_year, end_year in time_periods:
    # Filter the dataframes for the time period
    spei_df_period = pivot_df[(pivot_df.index >= start_year) & (pivot_df.index <= end_year)]
    usa_df_period = usa_df[(usa_df['Month'].dt.year >= start_year) & (usa_df['Month'].dt.year <= end_year)]
    
    # SPEI trend analysis
    trends = {}
    for state in spei_df_period.columns:
        X = np.arange(len(spei_df_period)).reshape(-1, 1)
        Y = spei_df_period[state].values
        X = sm.add_constant(X)
        mask = ~np.isnan(Y)
        X = X[mask]
        Y = Y[mask]
        model = sm.OLS(Y, X)
        results = model.fit()
        trends[state] = results.params[1]
    spei_trends_df = pd.DataFrame.from_dict(trends, orient='index', columns=['trend'])
    spei_trends_df = spei_trends_df.sort_values(by='trend')

    # Search Interest trend analysis
    trends = {}
    for (region, region_id), group in usa_df_period.groupby(['Region', 'Region ID']):
        year = group['Month'].dt.year.to_numpy().reshape(-1, 1)
        search_interest = group['Search Interest'].to_numpy().reshape(-1, 1)
        model = LinearRegression()
        model.fit(year, search_interest)
        trend = model.coef_[0][0]
        trends[region_id] = trend
    search_interest_trends = sorted(trends.items(), key=lambda x: x[1], reverse=True)[:20]
    search_interest_trends_df = pd.DataFrame(search_interest_trends, columns=['Region ID', 'Trend'])
    search_interest_trends_df['State'] = search_interest_trends_df['Region ID'].str.slice(start=3)

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

    # First plot
    sns.barplot(ax=axes[0], data=spei_trends_df.head(20).reset_index(), y='index', x='trend', palette='viridis')
    axes[0].set_title(f'Top 20 states with \n strongest decreasing SPEI trend ({start_year}-{end_year})', size = 15)
    axes[0].set_xlabel('SPEI trend \n(slope of regression line fitted to yearly data)', size = 15)
    axes[0].set_ylabel('State')
    axes[0].grid()

    # Second plot
    sns.barplot(ax=axes[1], data=search_interest_trends_df, y='State', x='Trend', palette='viridis')
    axes[1].set_title(f'Top 20 states with \n highest trend in Search Interest ({start_year}-{end_year})', size = 15)
    axes[1].set_xlabel('SI Trend \n(slope of regression line fitted to yearly data)', size = 15)
    axes[1].set_ylabel('State')
    axes[1].grid()

plt.tight_layout()
    
# Save the figure as a PNG image
plt.savefig(f"spei_si_trends.png", dpi=500)
    
# Save the figure as a pdf
plt.savefig(f"spei_si_trends.pdf")

    
plt.show()


# In[97]:


# Remove 'US-' from 'Region ID'
usa_df['Region ID'] = usa_df['Region ID'].str.replace('US-', '')

# Compute yearly average 'Search Interest' for each state
yearly_si_df = usa_df.groupby(['Year', 'Region ID'])['Search Interest'].mean().reset_index()

# Pivot the DataFrame to have years as rows and states as columns
pivot_si_df = yearly_si_df.pivot(index='Year', columns='Region ID', values='Search Interest')

# Compute the correlation between the yearly average SPEI and Search Interest
correlation_df = pivot_df.corrwith(pivot_si_df, axis=0)


# In[98]:


# Convert series to DataFrame for the heatmap
correlation_df = correlation_df.to_frame().reset_index()
correlation_df.columns = ['State', 'Correlation']

# Create a pivot table for the heatmap
correlation_pivot = correlation_df.pivot_table(index='State', values='Correlation')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_pivot, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation between Yearly SPEI and Google Trends Search Interest")
plt.show()


# In[104]:


# Sort by absolute value of correlation
correlation_df = correlation_df.reindex(correlation_df.Correlation.abs().sort_values(ascending=False).index)

# Barplot
plt.figure(figsize=(10, 10))
sns.barplot(x='Correlation', y='State', data=correlation_df)
plt.title("Correlation between Yearly SPEI and Google Trends Search Interest")
plt.grid()
plt.show()


# In[106]:


# Calculate trend for each state in pivot_df and pivot_si_df
spei_trends = pivot_df.apply(lambda x: np.polyfit(pivot_df.index, x, 1)[0], axis=0)
si_trends = pivot_si_df.apply(lambda x: np.polyfit(pivot_si_df.index, x, 1)[0], axis=0)

# Select states with a negative trend in SPEI
negative_spei_states = spei_trends[spei_trends < 0].index

# Among those states, select the ones with a positive trend in SI
selected_states = si_trends[negative_spei_states][si_trends[negative_spei_states] > 0].index

# Subset pivot_df and pivot_si_df for selected states
selected_spei = pivot_df[selected_states]
selected_si = pivot_si_df[selected_states]

# Flatten DataFrames and remove NaN values
spei_flattened = selected_spei.values.flatten()
si_flattened = selected_si.values.flatten()

# Keep only positions where neither is NaN
mask = ~np.isnan(spei_flattened) & ~np.isnan(si_flattened)
spei_flattened = spei_flattened[mask]
si_flattened = si_flattened[mask]

# Calculate overall correlation for selected states
selected_states_correlation = np.corrcoef(spei_flattened, si_flattened)[0, 1]
print(f"Correlation for states with decreasing SPEI and increasing SI: {selected_states_correlation:.2f}")

# Visualization
for state in selected_states:
    plt.figure(figsize=(10, 5))
    plt.title(f"SPEI vs SI for {state}")
    plt.plot(pivot_df.index, pivot_df[state], label="SPEI")
    plt.plot(pivot_si_df.index, pivot_si_df[state], label="SI")
    plt.legend()
    plt.show()


# In[119]:


fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for idx, state in enumerate(selected_states):
    row = idx // 4
    col = idx % 4
    ax1 = axes[row, col]
    ax2 = ax1.twinx()
    
    ax1.plot(pivot_si_df.index, pivot_si_df[state], label="SI", color='b')
    ax2.plot(pivot_df.index, pivot_df[state], label="SPEI", color='r')

    ax1.set_ylabel('SI', color='b')
    ax2.set_ylabel('SPEI', color='r')

    ax1.set_title(state, size = 20)
    ax1.grid(True)

    # Put a legend to the right of the current axis
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

# Place correlation and p-value information in the last subplot
axes[2, 3].axis('off')
axes[2, 3].text(0.5, 0.5, f"Correlation for states with\n decreasing SPEI and \n increasing SI: {correlation:.2f}\n"
                          f"P-value: {p_value:.2e}", 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                transform=axes[2, 3].transAxes)

plt.tight_layout()
plt.savefig('speidwn_siup_correlation.png', dpi = 400)
plt.show()


# In[109]:


correlation, p_value = pearsonr(spei_flattened, si_flattened)
print(f"Correlation for states with decreasing SPEI and increasing SI: {correlation:.2f}")
print(f"P-value: {p_value}")


# In[ ]:




