#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import fnmatch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, box
from shapely.ops import cascaded_union
# import rasterio.features
from mpl_toolkits.axes_grid1 import make_axes_locatable
import requests
from io import BytesIO
from PIL import Image

# Read the shapefile and filter non-CONUS states
states_gdf = gpd.read_file("cb_2020_us_state_20m.shp")
conus_states_gdf = states_gdf[~states_gdf["STUSPS"].isin(["AK", "HI", "AS", "GU", "MP", "PR", "VI"])]

image_shape = (720, 1080)

# List of model weight files to be loaded and evaluated
model_weight_files = [
    "model_weights_0ml.pth",
    "model_weights_1ml.pth",
    "model_weights_2ml.pth",
    "model_weights_3ml.pth",
    "model_weights_4ml.pth",
    "model_weights_5ml.pth"
]

# Load the saved R-squared array
r_squared_array = np.load("output/r_squared_array.npy", allow_pickle=True)
# state_r_squared = np.load("output/state_r_squared.npy", allow_pickle=True).item()

print(r_squared_array.shape)

def check_repeating_pattern(r_squared_array, num_models=6):
    for idx in range(0, len(r_squared_array), num_models):
        current_location = r_squared_array[idx][0]
        for offset in range(1, num_models):
            if r_squared_array[idx + offset][0] != current_location:
                return False
    return True

# Test if the r_squared_array has a repeating pattern of 6 tuples per pixel location
repeating_pattern = check_repeating_pattern(r_squared_array)

print("Repeating pattern:", repeating_pattern)


# # print(r_squared_array[:5])

# Create an empty array to store the maximum R-squared value for each pixel
max_r_squared = np.zeros(image_shape)
best_model_index = np.zeros(image_shape, dtype=int)



def geometry_to_mask(image_shape, geometry):
    height, width = image_shape
    bounds = geometry.bounds
    transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    shapes = [geometry]
    mask = rasterio.features.geometry_mask(shapes, transform=transform, out_shape=(height, width), invert=True)
    return mask



def crop_image_to_geometry(image, geometry):
    # Ensure the input image is 2-dimensional
    if len(image.shape) > 2:
        image = image.squeeze()
    elif len(image.shape) < 2:
        raise ValueError("Input image should be at least 2-dimensional.")

    # Convert the geometry to a binary mask
    mask = geometry_to_mask(image.shape, geometry)

    # Apply the mask to the image
    cropped_image = image * mask

    # Set the pixels outside the mask to zero
    cropped_image[~mask] = 0

    return cropped_image

def create_mask(image_shape, geometry):
    mask = geometry_to_mask(image_shape, geometry)
    return mask

# Create the mask for CONUS states
mask = create_mask(image_shape, conus_states_gdf.unary_union)

conus_bounds = conus_states_gdf.unary_union.bounds
minx, miny, maxx, maxy = conus_bounds



# Find the best models and the maximum R-squared values for each pixel
best_model_index, max_r_squared = find_best_models(r_squared_array)


# Create a masked array for the best model index
masked_best_model_index = np.ma.masked_array(best_model_index, mask=~mask)

# Plot the masked best model index heatmap
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
im = ax.imshow(masked_best_model_index, cmap='tab10', extent=[minx, maxx, miny, maxy], vmin=0, vmax=len(model_weight_files)-1)

# Plot the CONUS states
# conus_states_gdf.boundary.plot(ax=ax, linewidth=1, edgecolor="black")

# Set the axes limits
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# Remove the axes ticks
ax.set_xticks([])
ax.set_yticks([])

# Add a colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax, ticks=range(len(model_weight_files)))
cbar.ax.set_yticklabels(['Model 0', 'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'])

# Save the plot
plt.savefig("output/best_model_heatmap.png", dpi=600)

# Show the plot
plt.show()

# Close the plot
plt.close(fig)


In[ ]:


def find_best_models(r_squared_array, num_models=6):
    best_models = np.zeros(image_shape, dtype=int)
    max_r_squared = np.zeros(image_shape)
    
    for idx in range(0, len(r_squared_array), num_models):
        pixel_location = r_squared_array[idx][0]
        i, j = pixel_location
        max_r_square_value = r_squared_array[idx][1]
        best_model = 0
        
        for model_idx in range(1, num_models):
            r_square_value = r_squared_array[idx + model_idx][1]
            if r_square_value > max_r_square_value:
                max_r_square_value = r_square_value
                best_model = model_idx
        
        max_r_squared[i, j] = max_r_square_value
        best_models[i, j] = best_model
    
    return best_models, max_r_squared


# Find the best models and the maximum R-squared values for each pixel
best_model_index, max_r_squared = find_best_models(r_squared_array)

for item in r_squared_array:
    location, r_squared_value = item
    i, j = location
    model_index = idx
    if r_squared_value > max_r_squared[i, j]:
        max_r_squared[i, j] = r_squared_value
        best_model_index[i, j] = model_index

# Iterate over model weight files
for idx, model_weight_file in enumerate(model_weight_files):
    # Load the saved R-squared array for the current model
    r_squared_array = np.load(f"output/r_squared_array_{model_weight_file[:-4]}.npy", allow_pickle=True)

    # Create an empty R-squared heatmap
    r_squared_heatmap = np.zeros(image_shape)

    # Populate the heatmap with the R-squared values from the saved array
    for item in r_squared_array:
        location, r_squared_value = item
        i, j = location
        r_squared_heatmap[i, j] = r_squared_value

    # Update the best model index and maximum R-squared value for each pixel
    better_r_squared = r_squared_heatmap > max_r_squared
    best_model_index[better_r_squared] = idx
    max_r_squared[better_r_squared] = r_squared_heatmap[better_r_squared]




# In[ ]:




