#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import rasterio.features
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[ ]:


# Define paths to SPEI maps and Google trends maps
trends_dir = 'usa_maps_google'
spei_dir = 'usa_maps_spei'


# In[ ]:


def load_images_from_dir(directory):
    # Get a list of all image file names in the directory
    file_names = os.listdir(directory)
    # Load each image as a numpy array and add it to a list
    images = []
    for file_name in file_names:
        if file_name.endswith(".png"):
            file_path = os.path.join(directory, file_name)
            image = Image.open(file_path).convert('L')  # convert to grayscale
            # Resize the image
#             image = image.resize((target_width, target_height), Image.ANTIALIAS)
            image_array = np.array(image, dtype=np.float32)
            images.append(image_array)
    return images


# In[ ]:


# Load SPEI maps and Google trends maps

spei_maps = load_images_from_dir(spei_dir)

trends_maps = load_images_from_dir(trends_dir)


# In[ ]:


# Split images into train and test sets
spei_train, spei_test, trends_train, trends_test = train_test_split(spei_maps, trends_maps, test_size=0.2, random_state=42)


# In[ ]:


# def reshape_image(image):
#     if len(image.shape) > 2:
#         return image.squeeze()
#     return image


# In[ ]:


def geometry_to_mask(image_shape, geometry):
    height, width = image_shape
    bounds = geometry.bounds
    transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    shapes = [geometry]
    mask = rasterio.features.geometry_mask(shapes, transform=transform, out_shape=(height, width), invert=True)
    return mask


# In[ ]:


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


# In[ ]:


# def geometry_to_mask(image_shape, geometry):
#     # Create a shapely polygon from the input geometry
#     polygon = shape(geometry)
    
#     # Create an empty binary mask
#     height, width = image_shape
#     mask = np.zeros(image_shape, dtype=bool)
    
#     # Calculate the image bounds
#     image_bounds = box(0, 0, width, height)
    
#     # Calculate the intersection between the image bounds and the polygon
#     intersection = image_bounds.intersection(polygon)
    
#     # If the intersection is a MultiPolygon, convert it into a single Polygon
#     if intersection.type == 'MultiPolygon':
#         intersection = cascaded_union(intersection)
        
#     # Iterate through the pixels in the image
#     for y in range(height):
#         for x in range(width):
#             # Check if the current pixel is inside the intersection polygon
#             if intersection.contains(box(x, y, x + 1, y + 1)):
#                 mask[y, x] = True
                
#     return mask


# In[ ]:


# Read the shapefile and filter non-CONUS states
states_gdf = gpd.read_file("cb_2020_us_state_20m.shp")
conus_states_gdf = states_gdf[~states_gdf["STUSPS"].isin(["AK", "HI", "AS", "GU", "MP", "PR", "VI"])]


# In[ ]:


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder block 1 captures low-level features such as edges, corners, and textures in the input image.
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  # Convolution layer: input channels=1, output channels=64, kernel size=3x3, padding=1
            nn.ReLU(inplace=True),          # ReLU activation function (in-place to save memory)
            nn.Conv2d(64, 64, 3, padding=1), # Convolution layer: input channels=64, output channels=64, kernel size=3x3, padding=1
            nn.ReLU(inplace=True),          # ReLU activation function (in-place)
            nn.MaxPool2d(2, 2)              # Max-pooling layer: kernel size=2x2, stride=2
        )

        # Encoder block 2 responsible for capturing higher-level features and patterns in the input image.
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # Convolution layer: input channels=64, output channels=128, kernel size=3x3, padding=1
            nn.ReLU(inplace=True),            # ReLU activation function (in-place)
            nn.Conv2d(128, 128, 3, padding=1), # Convolution layer: input channels=128, output channels=128, kernel size=3x3, padding=1
            nn.ReLU(inplace=True),            # ReLU activation function (in-place)
            nn.MaxPool2d(2, 2)                # Max-pooling layer: kernel size=2x2, stride=2
        )
        
        # Middle block processes the high-level features captured by the encoder blocks.
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), # Convolution layer: input channels=128, output channels=256, kernel size=3x3, padding=1
            nn.ReLU(inplace=True),            # ReLU activation function (in-place)
            nn.Conv2d(256, 256, 3, padding=1), # Convolution layer: input channels=256, output channels=256, kernel size=3x3, padding=1
            nn.ReLU(inplace=True)             # ReLU activation function (in-place)
        )
        
        # Decoder block 1 starts the upsampling process to generate the output image. It combines the features from the middle block with the high-level features from encoder2.
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, 2, stride=2), # Transposed convolution layer (also called "deconvolution"): input channels=384, output channels=128, kernel size=2x2, stride=2
            nn.Conv2d(128, 128, 3, padding=1),         # Convolution layer: input channels=128, output channels=128, kernel size=3x3, padding=1
            nn.ReLU(inplace=True),                     # ReLU activation function (in-place)
            nn.Conv2d(128, 128, 3, padding=1),         # Convolution layer: input channels=128, output channels=128, kernel size=3x3, padding=1
            nn.ReLU(inplace=True),                     # ReLU activation function (in-place)
        )
        
        # Decoder block 2 continues the upsampling process and combines the features from the decoder1 block with the low-level features from encoder1.
        #It consists of a 2x2 transposed convolutional layer for upsampling, followed by two 3x3 convolutional layers, and a 1x1 convolutional layer to produce the final output image.
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 2, stride=2),  # Transposed convolution to upsample features
            nn.Conv2d(64, 64, 3, padding=1),          # 3x3 convolutional layer with 64 filters and padding of 1
            nn.ReLU(inplace=True),                    # ReLU activation (in-place to save memory)
            nn.Conv2d(64, 64, 3, padding=1),          # 3x3 convolutional layer with 64 filters and padding of 1
            nn.ReLU(inplace=True),                    # ReLU activation (in-place to save memory)
            nn.Conv2d(64, 1, 1)                       # 1x1 convolutional layer to produce the output map with a single channel
        )

    def forward(self, x):
        enc1 = self.encoder1(x)                      # Pass input through the first encoder block
        enc2 = self.encoder2(enc1)                   # Pass the output of the first encoder block through the second encoder block
        middle = self.middle(enc2)                   # Pass the output of the second encoder block through the middle block
        dec1 = self.decoder1(torch.cat((middle, enc2), 1))  # Concatenate the output of the middle and second encoder blocks, and pass through the first decoder block
        dec2 = self.decoder2(torch.cat((dec1, enc1), 1))    # Concatenate the output of the first decoder and first encoder blocks, and pass through the second decoder block
        return dec2                                  # Return the output of the second decoder block


# In[ ]:


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model instance
model = UNet()


# In[ ]:


# List of model weight files to be loaded and evaluated
model_weight_files = [
    "model_weights_0ml.pth",
    "model_weights_1ml.pth",
    "model_weights_2ml.pth",
    "model_weights_3ml.pth",
    "model_weights_4ml.pth",
    "model_weights_5ml.pth"
]


# In[ ]:


conus_bounds = conus_states_gdf.unary_union.bounds
minx, miny, maxx, maxy = conus_bounds


# In[ ]:


def create_mask(image_shape, geometry):
    mask = geometry_to_mask(image_shape, geometry)
    return mask

# Create the mask for CONUS states
image_shape = (720, 1080)
mask = create_mask(image_shape, conus_states_gdf.unary_union)


# In[ ]:


# Load the saved R-squared array
r_squared_array = np.load("output/r_squared_array.npy", allow_pickle=True)


# In[ ]:


# Iterate over model weight files
for model_weight_file in model_weight_files:
    # Create an empty R-squared heatmap
    image_shape = (720, 1080)
    r_squared_heatmap = np.zeros(image_shape)

    # Populate the heatmap with the R-squared values from the saved array
    for item in r_squared_array:
        location, r_squared_value = item
        i, j = location
        r_squared_heatmap[i, j] = r_squared_value

    # Create a masked array for the heatmap
    masked_r_squared_heatmap = np.ma.masked_array(r_squared_heatmap, mask=~mask)

    # Plot the masked R-squared heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    im = ax.imshow(masked_r_squared_heatmap, cmap='Blues', extent=[minx, maxx, miny, maxy], vmin=0, vmax=1)

    # Plot the CONUS states
#     conus_states_gdf.boundary.plot(ax=ax, linewidth=1, edgecolor="black")

    # Set the axes limits
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Remove the axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)

    # Save the plot
    plt.savefig(f"output/cropped_r_squared_heatmap_with_colorbar_{model_weight_file[:-4]}.png", dpi=600)
    
    # Show the plot
    plt.show()

    # Close the plot
    plt.close(fig)







