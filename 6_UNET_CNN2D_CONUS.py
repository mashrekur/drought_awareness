#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# print("Image dimensions spei:", spei_maps[0].shape)
# print("Image dimensions si:", trends_maps[0].shape)


# In[ ]:


# len(spei_maps) == len(trends_maps)


# In[ ]:


# Split images into train and test sets
spei_train, spei_test, trends_train, trends_test = train_test_split(spei_maps, trends_maps, test_size=0.2, random_state=42)


# In[ ]:


# print(len(spei_train))
# print(len(spei_test))
# print(len(trends_train))
# print(len(trends_test))


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


# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


# calculate the dimensions after the last pooling layer by running a dummy input through the convolutional layers
# dummy_input = torch.randn(1, 1, 1080, 720)
# conv_output = model.conv_layers(dummy_input)
# conv_output_shape = conv_output.size()
# print(conv_output_shape)


# In[ ]:


#with dense layers

num_epochs = 11

# Training loop
for epoch in range(num_epochs):
    model.train()
    for i in range(len(spei_train)):
        spei = torch.Tensor(spei_train[i]).unsqueeze(0).unsqueeze(0)
        trends = torch.Tensor(trends_train[i]).unsqueeze(0).unsqueeze(0)

        outputs = model(spei)
        loss = criterion(outputs, trends)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(spei_train)}], Loss: {loss.item():.4f}')

# Save the model weights after training
torch.save(model.state_dict(), "model_weights.pth")


# In[ ]:


# Testing loop
model.eval()
r2_scores = []
with torch.no_grad():
    for i in range(len(spei_test)):
        spei = torch.Tensor(spei_test[i]).unsqueeze(0).unsqueeze(0)
        trends = torch.Tensor(trends_test[i]).unsqueeze(0).unsqueeze(0)

        outputs = model(spei)
        r2 = r2_score(trends.cpu().numpy().flatten(), outputs.cpu().numpy().flatten())
        r2_scores.append(r2)

avg_r2_score = sum(r2_scores) / len(r2_scores)
print(f'Average R-squared score: {avg_r2_score:.4f}')
with open("r2_scores.txt", "w") as f:
    for r2 in r2_scores:
        f.write(f"{r2}\n")
    f.write(f"Average R-squared score: {avg_r2_score:.4f}\n")


# # Results:
# ### 0 month lag, num epoch = 20, Avg R-sq = 0.53
# ### 1 month lag, num epoch = 20, Avg R-sq = 0.5748
# ### 2 months lag, num epoch = 20, Avg R-sq = 0.6067
# ### 3 months lag, num epoch = 20, Avg R-sq = 0.6164
# ### 4 months lag, num epoch = 20, Avg R-sq = 0.5094
# ### 5 months lag, num epoch = 20, Avg R-sq = 0.5413
# 
# 

# In[6]:


# Function to read R-squared values from a file
def read_r2_values(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        r2_values = [float(line.strip()) for line in lines[:-1]]  # Exclude the last line
        avg_r2 = sum(r2_values) / len(r2_values)
    return r2_values, avg_r2


# In[7]:


file_names = ["r2_scores.txt", "r2_scores_1ml.txt", "r2_scores_2ml.txt", "r2_scores_3ml.txt", "r2_scores_4ml.txt", "r2_scores_5ml.txt"]
labels = ["0 months", "1 month", "2 months", "3 months", "4 months", "5 months"]
all_r2_values = []
average_r2_values = []


# In[8]:


for file_name in file_names:
    r2_values, avg_r2 = read_r2_values(file_name)
    all_r2_values.append(r2_values)
    average_r2_values.append(avg_r2)


# In[9]:


# Plot R-squared value distributions
fig, ax = plt.subplots()
print("Number of labels:", len(labels))
print("Number of data columns:", len(all_r2_values))
ax.boxplot(all_r2_values, labels=labels)
ax.scatter(range(1, len(average_r2_values) + 1), average_r2_values, color='r', label='Average R-squared', zorder=3)

plt.xlabel("Lag")
plt.ylabel("R-squared values")
plt.title("R-squared value distributions for different lags")
plt.legend()
plt.grid()
plt.show()


# In[ ]:




