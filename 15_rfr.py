#!/usr/bin/env python
# coding: utf-8

# In[641]:


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
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import weibull_min
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.model_selection import KFold
from scipy.stats import skew, kurtosis, norm
from scipy.stats import weibull_min, skew, kurtosis
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns


# In[2]:


# Load the US states shapefile
us_states = gpd.read_file('cb_2018_us_state_500k.shp')


# In[361]:


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


# In[362]:


#Make a list of dates to iterate over and generate maps
dates = spei_df.columns[2:].to_list()


# Load data into a geopandas dataframe
gdf = gpd.GeoDataFrame(spei_df['2020-12-16'], geometry=[Point(xy) for xy in zip(spei_df.lon, spei_df.lat)])


# Load the US states shapefile
us_map = gpd.read_file('cb_2018_us_state_500k.shp')

# Filter out non-CONUS states from us_map
us_map = us_map[~us_map['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico', 'United States Virgin Islands'])]

# Correctly filter spei_df to retain 'lat', 'lon', and dates within 2004-2020
columns_to_keep = ['lat', 'lon'] + [col for col in spei_df.columns if '2004-01-01' <= col <= '2020-12-31']
spei_df_filtered = spei_df[columns_to_keep]

# Create a GeoDataFrame from the filtered SPEI DataFrame
gdf_spei = gpd.GeoDataFrame(
    spei_df_filtered,
    geometry=gpd.points_from_xy(spei_df_filtered.lon, spei_df_filtered.lat)
)
gdf_spei.crs = "EPSG:4326"  # Set the coordinate reference system to WGS84


spei_states = sjoin(gdf_spei, us_map, how="inner", op='intersects')

# Drop non-SPEI columns to focus on SPEI values
spei_values_columns = [col for col in spei_states.columns if '2004-01-01' <= col <= '2020-12-31']


# In[363]:


# Calculate the mean SPEI for each row (pixel) across all time steps
spei_states['avg_spei'] = spei_states[spei_values_columns].mean(axis=1)

# Group by state NAME and calculate the average SPEI per state
state_avg_spei = spei_states.groupby('NAME')['avg_spei'].mean().reset_index()


# In[364]:


spei_columns = spei_states.columns[2:-12]

# Melt the DataFrame to have 'lat', 'lon', 'NAME', and 'SPEI' values in long format
spei_long_df = spei_states.melt(id_vars=['lat', 'lon', 'NAME'], var_name='date', value_name='SPEI', value_vars=spei_columns)


# Filter out non-SPEI and non-state name columns
spei_long_df = spei_long_df[['NAME', 'SPEI']]

# Group by state NAME to get SPEI values for each state
grouped_spei = spei_long_df.groupby('NAME')


# In[365]:


spei_states


# In[368]:


# Plotting
n_states = len(grouped_spei)
ncols = 2
nrows = n_states // ncols + (n_states % ncols > 0)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3), tight_layout=True)
axes = axes.flatten()

for i, (state, data) in enumerate(grouped_spei):
    ax = axes[i]
    data['SPEI'].hist(bins=20, ax=ax, alpha=0.7, label=f'{state} SPEI', color='skyblue', edgecolor='black')
    ax.set_title(f'{state} SPEI Distribution')
    ax.set_xlabel('SPEI Value')
    ax.set_ylabel('Frequency')
    ax.legend()

# Hide any unused subplots if the number of states is odd
if n_states % ncols != 0:
    for j in range(i + 1, nrows * ncols):
        axes[j].set_visible(False)
plt.savefig('h1_spei.png', bbox_inches='tight')
plt.show()


# In[369]:


# Convert dates to a numeric time scale (e.g., months from start)
dates = pd.to_datetime(spei_columns)
months_from_start = (dates - dates.min()) / np.timedelta64(1, 'M')

# Prepare the independent variable (time) for linear regression
X = months_from_start.values.reshape(-1, 1)

# Initialize a DataFrame to store trends
trend_df = spei_states[['lat', 'lon', 'NAME']].copy().drop_duplicates()
trend_df['trend'] = np.nan

for i, row in trend_df.iterrows():
    # Extract SPEI values for the current location
    y = spei_states.loc[(spei_states['lat'] == row['lat']) & (spei_states['lon'] == row['lon']), spei_columns].values.flatten()
    if not np.isnan(y).all():  # Check if there are enough data points
        # Fit linear regression and extract the slope (trend)
        model = LinearRegression().fit(X, y)
        trend_df.at[i, 'trend'] = model.coef_[0]

# Drop rows where trend could not be calculated
trend_df.dropna(subset=['trend'], inplace=True)


# In[370]:


# Group by state NAME to get trend values for each state
grouped_trends = trend_df.groupby('NAME')

# Plotting
n_states = len(grouped_trends)
ncols = 2
nrows = n_states // ncols + (n_states % ncols > 0)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*4), tight_layout=True)
axes = axes.flatten()

for i, (state, data) in enumerate(grouped_trends):
    ax = axes[i]
    data['trend'].hist(bins=20, ax=ax, alpha=0.7, label=f'{state} Trend', color='skyblue', edgecolor='black')
    ax.set_title(f'{state} SPEI Trend Distribution')
    ax.set_xlabel('Trend (SPEI change per month)')
    ax.set_ylabel('Frequency')
    ax.legend()

# Hide any unused subplots if the number of states is odd
if n_states % ncols != 0:
    for j in range(i + 1, nrows * ncols):
        axes[j].set_visible(False)
plt.savefig('h2_trend_spei.png', bbox_inches='tight')
plt.show()


# In[371]:


# Initialize a DataFrame to store variances
variance_df = spei_states[['lat', 'lon', 'NAME']].copy().drop_duplicates()
variance_df['variance'] = np.nan

for i, row in variance_df.iterrows():
    # Extract SPEI values for the current location
    spei_values = spei_states.loc[(spei_states['lat'] == row['lat']) & (spei_states['lon'] == row['lon']), spei_columns]
    # Calculate variance of SPEI values
    variance = spei_values.var(axis=1, skipna=True)  # skipna=True ensures that NaNs are ignored
    # Assign variance to the DataFrame
    if not variance.empty:
        variance_df.at[i, 'variance'] = variance.values[0]

# Drop rows where variance could not be calculated
variance_df.dropna(subset=['variance'], inplace=True)


# In[372]:


# Group by state NAME to get variance values for each state
grouped_variances = variance_df.groupby('NAME')

# Plotting
n_states = len(grouped_variances)
ncols = 2
nrows = n_states // ncols + (n_states % ncols > 0)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*4), tight_layout=True)
axes = axes.flatten()

for i, (state, data) in enumerate(grouped_variances):
    ax = axes[i]
    data['variance'].hist(bins=20, ax=ax, alpha=0.7, label=f'{state} Variance', color='skyblue', edgecolor='black')
    ax.set_title(f'{state} SPEI Variance Distribution')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Frequency')
    ax.legend()

# Hide any unused subplots if the number of states is odd
if n_states % ncols != 0:
    for j in range(i + 1, nrows * ncols):
        axes[j].set_visible(False)
plt.savefig('h3_variance.png', bbox_inches='tight')
plt.show()


# In[373]:


spei_states.columns


# In[374]:


spei_columns = spei_states.columns[2:-15].tolist()
# spei_columns

# Define drought threshold
drought_threshold = -1.0

def calculate_continuous_drought_lengths(row):
    drought_lengths = []
    current_length = 0
    
    for col in spei_columns:
        value = row[col]
        if value < drought_threshold:
            current_length += 1
        elif current_length > 0:
            drought_lengths.append(current_length)
            current_length = 0
    # Catch any ongoing drought period at the end of the data
    if current_length > 0:
        drought_lengths.append(current_length)
        
    return drought_lengths

# Apply the function to each row and store the result in a new column
spei_states['DroughtLengths'] = spei_states.apply(calculate_continuous_drought_lengths, axis=1)

# Aggregate drought lengths by state
def aggregate_drought_lengths(drought_lengths_list):
    aggregated_lengths = {}
    for lengths in drought_lengths_list:
        for length in lengths:
            if length in aggregated_lengths:
                aggregated_lengths[length] += 1
            else:
                aggregated_lengths[length] = 1
    return aggregated_lengths

# Group by state and aggregate
drought_length_aggregates = spei_states.groupby('NAME')['DroughtLengths'].apply(aggregate_drought_lengths).to_dict()


# In[375]:


organized_data = {}
for (state, length), frequency in drought_length_aggregates.items():
    if pd.isna(frequency):  # Skip NaN values
        continue
    if state not in organized_data:
        organized_data[state] = {}
    organized_data[state][length] = frequency

# Determine the number of subplots needed
n_states = len(organized_data)
nrows = int(np.ceil(n_states / 2))  # 2 columns of subplots
fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 4 * nrows), squeeze=False)
axes_flat = axes.flatten()

# Plotting
for idx, (state, lengths_dict) in enumerate(organized_data.items()):
    lengths = list(lengths_dict.keys())
    frequencies = list(lengths_dict.values())
    axes_flat[idx].bar(lengths, frequencies, color='skyblue')
    axes_flat[idx].set_title(f'Drought Length Frequency for {state}')
    axes_flat[idx].set_xlabel('Drought Length (months)')
    axes_flat[idx].set_ylabel('Frequency')
    axes_flat[idx].set_xlim(0, 8)  # Set the x-axis to go from 0 to 7
    axes_flat[idx].set_xticks(np.arange(0, 9, 1))  # Set ticks at every 1 month interval

# Hide unused axes if the total number of states is odd
if n_states % 2:
    axes_flat[-1].axis('off')

plt.tight_layout()
plt.savefig('h4_drought_length.png', bbox_inches='tight')
plt.show()


# In[376]:


# Step 1: Identify Drought Occurrences and Calculate Mean SPEI
def calculate_mean_drought_severity(row, spei_columns):
    drought_values = [row[col] for col in spei_columns if row[col] < -1.0]
    if drought_values:
        return np.mean(drought_values)  # Return mean SPEI for drought months
    else:
        return np.nan  # Return NaN if no drought months


# Calculate mean drought severity for each data point
spei_states['MeanDroughtSeverity'] = spei_states.apply(calculate_mean_drought_severity, spei_columns=spei_columns, axis=1)

# Step 2: Aggregate Mean Drought Severity by State
state_mean_severity = spei_states.groupby('NAME')['MeanDroughtSeverity'].apply(list).to_dict()

# Step 3: Plot Histograms for Each State
fig, axes = plt.subplots(nrows=int(np.ceil(len(state_mean_severity)/3)), ncols=3, figsize=(15, 4 * np.ceil(len(state_mean_severity)/3)))
axes_flat = axes.flatten()

for i, (state, severity_list) in enumerate(state_mean_severity.items()):
    # Clean the list by removing NaN values
    clean_severity_list = [x for x in severity_list if not np.isnan(x)]
    if clean_severity_list:  # Only plot if there are drought occurrences
        axes_flat[i].hist(clean_severity_list, bins=20, color='skyblue', edgecolor='black')
        axes_flat[i].set_title(state)
        axes_flat[i].set_xlabel('Mean Drought Severity (SPEI)')
        axes_flat[i].set_ylabel('Frequency')
    else:
        axes_flat[i].set_visible(False)  # Hide axes without data

# Adjust layout and show plot
plt.tight_layout()
plt.savefig('h5_magnitude_spei.png', bbox_inches='tight')
plt.show()


# In[378]:


# Let's create a dictionary to hold various statistics for each state
spei_stats_params = {}

for state, data in grouped_spei:
    # Extract SPEI values for the state
    spei_values = data['SPEI']
    
    # Fit a normal distribution to the SPEI data and calculate additional statistics
    mu, std = norm.fit(spei_values)
    skewness = skew(spei_values)
    kurt = kurtosis(spei_values)
    percentile_25 = np.percentile(spei_values, 25)
    median = np.percentile(spei_values, 50)  # 50th percentile is the median
    percentile_75 = np.percentile(spei_values, 75)
    coefficient_of_variation = std / mu if mu != 0 else np.nan  # Handle division by zero
    data_range = np.max(spei_values) - np.min(spei_values)
    iqr = np.percentile(spei_values, 75) - np.percentile(spei_values, 25)
    
    # Store the calculated parameters
    spei_stats_params[state] = {
        'mean': mu,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurt,
        'percentile_25': percentile_25,
        'median': median,
        'percentile_75': percentile_75,
        'coefficient_of_variation': coefficient_of_variation,
        'range': data_range,
        'IQR': iqr
    }

# `spei_stats_params` now contains the desired statistics for the SPEI values of each state


# In[380]:


spei_stats_params


# In[381]:


trend_stats_params = {}

for state, data in grouped_trends:
    # Extract trend values for the state
    trend_values = data['trend']
    
    # Fit a normal distribution to the trend data and calculate additional statistics
    mu, std = norm.fit(trend_values)
    skewness = skew(trend_values)
    kurt = kurtosis(trend_values)
    percentile_25 = np.percentile(trend_values, 25)
    median = np.percentile(trend_values, 50)  # 50th percentile is the median
    percentile_75 = np.percentile(trend_values, 75)
    coefficient_of_variation = std / mu if mu != 0 else np.nan  # Handle division by zero
    data_range = np.max(trend_values) - np.min(trend_values)
    iqr = np.percentile(trend_values, 75) - np.percentile(trend_values, 25)
    
    # Store the calculated parameters
    trend_stats_params[state] = {
        'mean': mu,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurt,
        'percentile_25': percentile_25,
        'median': median,
        'percentile_75': percentile_75,
        'coefficient_of_variation': coefficient_of_variation,
        'range': data_range,
        'IQR': iqr
    }


# In[383]:


trend_stats_params


# In[384]:


# `grouped_variances` is a DataFrame grouped by state with a column 'variance'
variance_stats_params = {}

for state, data in grouped_variances:
    # Extract variance values for the state
    variance_values = data['variance']
    
    # Fit a normal distribution to the variance data and calculate additional statistics
    mu, std = norm.fit(variance_values)
    skewness = skew(variance_values)
    kurt = kurtosis(variance_values)
    percentile_25 = np.percentile(variance_values, 25)
    median = np.percentile(variance_values, 50)  # 50th percentile is the median
    percentile_75 = np.percentile(variance_values, 75)
    coefficient_of_variation = std / mu if mu != 0 else np.nan  # Handle division by zero
    data_range = np.max(variance_values) - np.min(variance_values)
    iqr = np.percentile(variance_values, 75) - np.percentile(variance_values, 25)
    
    # Store the calculated parameters
    variance_stats_params[state] = {
        'mean': mu,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurt,
        'percentile_25': percentile_25,
        'median': median,
        'percentile_75': percentile_75,
        'coefficient_of_variation': coefficient_of_variation,
        'range': data_range,
        'IQR': iqr
    }


# In[385]:


variance_stats_params


# In[339]:


drought_length_fit_params = {}

for (state, length), frequency in drought_length_aggregates.items():
    if pd.isna(frequency):
        continue  # Skip this entry if the frequency is NaN
    if state not in drought_length_fit_params:
        drought_length_fit_params[state] = []
    # Ensure frequency is an integer
    frequency_int = int(frequency) if not pd.isna(frequency) else 0
    drought_length_fit_params[state].extend([length] * frequency_int)
    
drought_length_poisson_params = {}

for state, drought_lengths in drought_length_fit_params.items():
    # Calculate the mean (lambda) of the observed drought lengths
    lambda_hat = np.mean(drought_lengths)
    drought_length_poisson_params[state] = {'lambda': lambda_hat}
    
#     loc, lambda_mle = poisson.fit(drought_lengths, floc=0)  # floc=0 fixes the location parameter at 0
#     drought_length_poisson_params[state]['lambda_mle'] = lambda_mle


# Now, `drought_length_fit_params` is a dictionary with states as keys and a list of drought lengths as values,
# where each drought length is repeated according to its frequency.


# In[340]:


drought_length_poisson_params.items()


# In[387]:


#  drought_length_aggregates is structured with (state, length) as keys and frequency as values
for state, drought_lengths in drought_length_fit_params.items():
    if drought_lengths:  # Ensure the list is not empty
        # Existing Weibull fitting
        shape, loc, scale = weibull_min.fit(drought_lengths, floc=0)
        
        # Calculate additional statistics
        mean_length = np.mean(drought_lengths)
        variance_length = np.var(drought_lengths)
        max_length = np.max(drought_lengths)
        min_length = np.min(drought_lengths)
        percentile_25 = np.percentile(drought_lengths, 25)
        median_length = np.percentile(drought_lengths, 50)  # Median is the 50th percentile
        percentile_75 = np.percentile(drought_lengths, 75)
        skewness = skew(drought_lengths)
        kurt = kurtosis(drought_lengths)
        
        # Store the calculated parameters and statistics
        drought_length_weibull_params[state] = {
            'shape': shape,
            'scale': scale,
            'mean': mean_length,
            'variance': variance_length,
            'max': max_length,
            'min': min_length,
            'percentile_25': percentile_25,
            'median': median_length,
            'percentile_75': percentile_75,
            'skewness': skewness,
            'kurtosis': kurt
        }
    else:
        # Handle empty or missing data
        drought_length_weibull_params[state] = {key: None for key in ['shape', 'scale', 'mean', 'variance', 'max', 'min', 'percentile_25', 'median', 'percentile_75', 'skewness', 'kurtosis']}


# In[388]:


drought_length_weibull_params


# In[389]:


severity_stats_params = {}

for state, severity_list in state_mean_severity.items():
    # Clean the list by removing NaN values
    clean_severity_list = [x for x in severity_list if not np.isnan(x)]
    
    # Fit a normal distribution to the clean mean drought severity data and calculate additional statistics
    if clean_severity_list:  # Ensure the list is not empty
        mu, std = norm.fit(clean_severity_list)
        skewness = skew(clean_severity_list)
        kurt = kurtosis(clean_severity_list)
        percentile_25 = np.percentile(clean_severity_list, 25)
        median = np.percentile(clean_severity_list, 50)  # 50th percentile is the median
        percentile_75 = np.percentile(clean_severity_list, 75)
        coefficient_of_variation = std / mu if mu != 0 else np.nan  # Handle division by zero
        data_range = np.max(clean_severity_list) - np.min(clean_severity_list)
        iqr = np.percentile(clean_severity_list, 75) - np.percentile(clean_severity_list, 25)
        
        # Store the calculated parameters
        severity_stats_params[state] = {
            'mean': mu,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurt,
            'percentile_25': percentile_25,
            'median': median,
            'percentile_75': percentile_75,
            'coefficient_of_variation': coefficient_of_variation,
            'range': data_range,
            'IQR': iqr
        }
    else:
        # Handle empty or missing data
        severity_stats_params[state] = {'mean': None, 'std': None, 'skewness': None, 'kurtosis': None, 'percentile_25': None, 'median': None, 'percentile_75': None, 'coefficient_of_variation': None, 'range': None, 'IQR': None}


# In[390]:


severity_stats_params


# In[391]:


# Create DataFrames from the updated parameters
spei_df = pd.DataFrame.from_dict(spei_stats_params, orient='index').rename(columns=lambda x: f'SPEI_{x}')
trend_df = pd.DataFrame.from_dict(trend_stats_params, orient='index').rename(columns=lambda x: f'Trend_{x}')
variance_df = pd.DataFrame.from_dict(variance_stats_params, orient='index').rename(columns=lambda x: f'Variance_{x}')
drought_length_df = pd.DataFrame.from_dict(drought_length_weibull_params, orient='index').rename(columns=lambda x: f'DroughtLength_{x}')
severity_df = pd.DataFrame.from_dict(severity_stats_params, orient='index').rename(columns=lambda x: f'Severity_{x}')

# Combine all DataFrames into a single feature space DataFrame
feature_space_df = pd.concat([spei_df, trend_df, variance_df, drought_length_df, severity_df], axis=1)


# In[629]:


# feature_space_df 


# In[393]:


# Load the overall average R-squared values
with open('overall_state_averages.json', 'r') as file:
    overall_state_averages = json.load(file)

# Convert the loaded JSON data into a DataFrame
overall_state_averages_df = pd.DataFrame(list(overall_state_averages.items()), columns=['NAME', 'Overall_R_squared'])

# Merge the GeoDataFrame with the overall state averages DataFrame
overall_avg_state_model_df = us_states.merge(overall_state_averages_df, on='NAME')


# In[394]:


overall_avg_state_model_df_sorted = overall_avg_state_model_df.sort_values(by='NAME')
# overall_avg_state_model_df_sorted


# In[395]:


# Copy the 'Overall_R_squared' column from overall_avg_state_model_df to feature_space_df
feature_space_df['Overall_R_squared'] = overall_avg_state_model_df_sorted['Overall_R_squared'].values


# In[630]:


# feature_space_df


# In[617]:


# Columns to be dropped
columns_to_drop = [
#     'Trend_mean', 
    'Severity_coefficient_of_variation',
    'Severity_median',
    'Severity_mean',
#     'Variance_std',
#     'DroughtLength_scale',
    'Variance_coefficient_of_variation',
    'DroughtLength_variance',  
    'Variance_range',
    'SPEI_IQR',
    'Trend_percentile_25',
    'Variance_mean',
    'SPEI_std',
    'DroughtLength_mean',
    'Severity_percentile_75',
    'Trend_median',
    'DroughtLength_max',
    'DroughtLength_percentile_75',
    'DroughtLength_median',
    'DroughtLength_percentile_25',
    'DroughtLength_min',
    'SPEI_skewness',
    'SPEI_kurtosis',
    'SPEI_coefficient_of_variation',
#     'SPEI_range',
    'SPEI_percentile_25',
    'SPEI_percentile_75',
    'Trend_skewness',
#     'Trend_kurtosis',
    'Trend_coefficient_of_variation',
    'Trend_range',
    'Trend_percentile_75',
    'Variance_skewness',
#     'Variance_kurtosis',
    'Variance_coefficient_of_variation',
    'Variance_IQR',
    'DroughtLength_skewness',
    'DroughtLength_kurtosis',
    'Severity_skewness',
    'Severity_kurtosis',
#     'Severity_range',
    'Severity_IQR',
    'Variance_percentile_75',
    'Variance_percentile_25',
    'Severity_percentile_25',
    'DroughtLength_scale',
    'DroughtLength_shape',
#     'Trend_std',
    'Variance_median',
    'Variance_std'
]
# Drop the specified columns from the feature space DataFrame
feature_space = feature_space_df.drop(columns=columns_to_drop)


# In[618]:


# X = feature_space_df.iloc[:, :-1]  # All rows, all columns except the last one
# y = feature_space_df.iloc[:, -1]   # All rows, just the last column


# In[631]:


X = feature_space.iloc[:, :-1]  # All rows, all columns except the last one
y = feature_space.iloc[:, -1]   # All rows, just the last column


# In[620]:


# Initialize the Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=150, random_state=42)

# Fit the model to the entire dataset
rfr.fit(X, y)


# In[639]:


bootstrap_iterations = 500  
r2_scores = []
mse_scores = []

for _ in range(bootstrap_iterations):
    # create a bootstrap sample of features with replacement
    X_sample, y_sample = resample(X, y)
    
    # fit RFR on the sample
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_sample, y_sample)
    
    # predict on the original dataset 
    y_pred = model.predict(X)
    
    # calculate r2 score and mse 
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    r2_scores.append(r2)
    mse_scores.append(mse)

# calculate the average r2 score and mse overall
average_r2_score = np.mean(r2_scores)
average_mse_score = np.mean(mse_scores)

# calculate confidence intervals for r2 and mse
r2_confidence_interval = np.percentile(r2_scores, [2.5, 97.5])
mse_confidence_interval = np.percentile(mse_scores, [2.5, 97.5])

print(f"Average r2 Score: {average_r2_score}")
print(f"r2 Confidence Interval: {r2_confidence_interval}")
print(f"Average MSE Score: {average_mse_score}")
print(f"MSE Confidence Interval: {mse_confidence_interval}")


# In[640]:


# Initialize lists to store OOB scores
oob_r2_scores = []
oob_mse_scores = []

for _ in range(500):
    # Create a bootstrap sample of the dataset with replacement
    # Note: For OOB evaluation, the entire dataset is used for training, and OOB samples are automatically used for testing
    model = RandomForestRegressor(n_estimators=150, random_state=42, oob_score=True)
    model.fit(X, y)
    
    # OOB predictions are not directly available, so we use the OOB score for R2
    # Since mean squared error is not directly provided as an OOB metric, we focus on the R2 score for OOB evaluation
    oob_r2 = model.oob_score_  # This is the R2 score estimated using OOB samples
    
    # Append the OOB R2 score to the list
    oob_r2_scores.append(oob_r2)

# Since we're using OOB samples for evaluation, we don't directly calculate MSE here
# Calculate the average OOB R² score across all bootstrap iterations
average_oob_r2_score = np.mean(oob_r2_scores)

# Calculate confidence intervals for the OOB R² score
# oob_r2_confidence_interval = np.percentile(oob_r2_scores, [2.5, 97.5])

print(f"Average OOB R² Score: {average_oob_r2_score}")
# print(f"OOB R² Confidence Interval: {oob_r2_confidence_interval}")


# In[636]:


# Get and print the feature importances
feature_importances = rfr.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

print(importance_df.sort_values(by='Importance', ascending=False))


# In[649]:


# Create a dataframe for feature importances
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(15, 12))
plt.rcParams.update({'font.size': 16})
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title(f'Feature Importances of Random Forest Regression \nR-squared(10-fold x-val): 0.20, MSE: 0.009')
plt.xlabel('Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.grid()
sns.set_style("whitegrid")
plt.savefig('rfr_feature.png')
plt.show()


# In[650]:


importance_df


# In[675]:


num_vars = len(importance_df)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() # Create even segments for each feature
# Ensure the plot is a closed loop
angles += angles[:1]  # Complete the loop

values = importance_df['Importance'].tolist()
values += values[:1]  # Ensure the values loop back to the start value

# Plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Draw one axe per variable + add labels
ax.set_thetagrids(np.degrees(angles[:-1]), importance_df['Feature'])

# Draw ylabels - use percentages here for better interpretability
ax.set_rlabel_position(40)  # Move radial labels away from plotted line
# Define the range of yticks based on data's range
max_importance = max(importance_df['Importance']) * 100
yticks = np.linspace(0, max_importance, 5)  # Creates 5 evenly spaced ticks
plt.yticks(yticks, [f'{y:.0f}%' for y in yticks], color="black", size=14)
plt.ylim(0, )

# Change the grid properties - lighter grid lines, more discernible labels
ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Plot data - scale up the values to percentages
scaled_values = [val * 100 for val in values]
ax.plot(angles, scaled_values, linewidth=2, linestyle='solid', label='Importance')
ax.fill(angles, scaled_values, 'red', alpha=0.3)

# Add dots at each plot point
ax.scatter(angles[:-1], scaled_values[:-1], color='black', s=50)  # 's' adjusts the size of the dots

# Add a title with increased font size
plt.title(f'Feature Importances of Random Forest Regression \nR-squared(10-fold x-val): 0.20, MSE: 0.009 \n', size=18)
plt.savefig('rfr_feature.png')
plt.show()


# In[624]:


# X and y are already defined
kf = KFold(n_splits=10, shuffle=True, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42)

# Lists to store observed and predicted values for all folds
all_observed = []
all_predicted = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Append observed and predicted values for this fold to the lists
    all_observed.extend(y_test)
    all_predicted.extend(y_pred)

# Convert lists to numpy arrays for performance metrics calculation
all_observed = np.array(all_observed)
all_predicted = np.array(all_predicted)

# Calculate R² and MSE for all data
overall_r2 = r2_score(all_observed, all_predicted)
overall_mse = mean_squared_error(all_observed, all_predicted)

print(f"Overall R² for all data: {overall_r2}")
print(f"Overall MSE for all data: {overall_mse}")


# In[635]:


# X = feature_space_df.iloc[:, :-1]  # All rows, all columns except the last one
# y = feature_space_df.iloc[:, -1]   # All rows, just the last column


kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=150, random_state=42)

# Lists to store observed and predicted values for all folds
all_observed = []
all_predicted = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Append observed and predicted values for this fold to the lists
    all_observed.extend(y_test)
    all_predicted.extend(y_pred)

# Convert lists to numpy arrays for performance metrics calculation
all_observed = np.array(all_observed)
all_predicted = np.array(all_predicted)

# Calculate R² and MSE for all data
overall_r2 = r2_score(all_observed, all_predicted)
overall_mse = mean_squared_error(all_observed, all_predicted)

print(f"Overall R² for all data: {overall_r2}")
print(f"Overall MSE for all data: {overall_mse}")


# In[458]:





# In[ ]:




