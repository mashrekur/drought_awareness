#!/usr/bin/env python
# coding: utf-8

# In[10]:


import hiddenlayer as hl
import torch
import torch.nn as nn
import numpy as np
from IPython.display import Image, display


# In[11]:


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
        return dec2               


# In[12]:


# Instantiate the model
model = UNet()


# In[15]:


hl_graph = hl.build_graph(model, torch.zeros([1, 1, 128, 128]))
hl_graph.theme = hl.graph.THEMES["blue"].copy()  # Set the theme of the graph

dot = hl_graph.build_dot()
dot.attr(rankdir="LR")  # Set the rankdir attribute to 'LR' after creating the Digraph object
dot.render("./unet_model", format="png", cleanup=True) # Save the graph as a PNG image
# dot.view()  # Open the graph in a separate window for easier viewing
display(Image(filename="./unet_model.png")) # Display the saved image in the notebook


# In[ ]:




