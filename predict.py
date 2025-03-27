from scipy.stats import wasserstein_distance

import os
import cv2
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import os
import altair as alt
import shap
import torch
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

truemean = pd.read_csv('attribute_means.csv')

import time
import pickle
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--var', type=str, default=None)
args = argparser.parse_args()

finished_attributes = [
    "trustworthy", "attractive", "dominant", "smart", "age", "gender", "weight", 
    "typical", "happy", "familiar", "outgoing", "memorable", "well-groomed", 
    "long-haired", "smug", "dorky", "skin-color", "hair-color", "alert", "cute", 
    "privileged", "liberal", "asian", "middle-eastern", "hispanic", "islander", 
    "native", "black", "white", "looks-like-you", "gay", "electable", "godly", 
    "outdoors"
]

results = []
for fa in finished_attributes:
    column = fa
    transform = "resnet"
    seed = 0
    var = args.var
    if var == 'trashtrue':
        trash = True
    elif var == 'trashfalse':
        trash = False
    else:
        trash = None

    if var == 'transfer' or var == 'initial': 
        weight_path = f"weights/{var}/trained_epoch100_attribute{column}_transform{transform}_weights.pth"
    else:
        weight_path = f"weights/trained_epoch100_trash{trash}_attribute{column}_transform{transform}_weights.pth"

    # Seed and Device Initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    truemean = pd.read_csv('attribute_means.csv')

    class OMI_Dataset(Dataset):
        def __init__(self, folder_path, var=var, column=column, transform=transform):
            """
            Custom dataset for loading images and their ratings
            
            Args:
            - folder_path (str): Path to the train_set folder
            - transform (callable, optional): Optional image transformations
            """
            self.folder_path = folder_path
            self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.column = column
            self.var = var

            # Transform for preprocessing images
            if transform == 'resnet':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
                ])
            elif transform == 'none':
                self.transform = transforms.Compose([
                    transforms.ToTensor()  # Only converts to tensor, no transformation
                ])
            else:
                raise ValueError(f"Invalid transform: {transform}")
        
        def true_mean(self, image_name):
            """
            Calculate the true mean rating for a given image
            
            Args:
            - image_name (str): Name of the image file
            
            Returns:
            - float: Ratingscore between 1 and 100
            """
            stim = str(image_name)
            if '.' in stim:
                stim = int(stim.split('.')[0])  # Split by '.' and take the first part
            else:
                stim = int(stim)
            return truemean[self.column][stim - 1]  # For i+1.jpg, should look for truemean['happy'][i]
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            image_name = self.image_files[idx]
            image_path = os.path.join(self.folder_path, image_name)
            
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            # Get true mean rating OR constant label
            if self.var == "all50":
                label = 50
            else:
                label = self.true_mean(image_name)
            
            return image, torch.tensor(label, dtype=torch.float32)

    # Create test dataloader
    test_dataset = OMI_Dataset('test_set')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = resnet50()

    # Modify the model's final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

    loaded_model = model
    loaded_model.load_state_dict(torch.load(weight_path))
    loaded_model.to(device)
    loaded_model.eval()

    # Disable gradient computation for inference
    with torch.no_grad():
        total_mse = 0.0
        num_batches = 0
        
        for test_images, test_labels in test_dataloader:
            # Move data to device
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            
            # Forward pass
            outputs = loaded_model(test_images)
            
            # Compute MSE for this batch
            batch_mse = nn.MSELoss()(outputs.squeeze(), test_labels)
            total_mse += batch_mse.item()
            num_batches += 1

            if num_batches == 1:
                mse_2img = nn.MSELoss()(outputs.squeeze()[:2], test_labels[:2]).item()
        
        # Calculate average MSE across all batches
        average_mse = total_mse / num_batches
    
        results.append({
            "attribute": column,
            "mse_2img": mse_2img,
            "average_mse": average_mse
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv(f'performance/{args.var}_mse_results.csv', index=False)
print("Results saved", f'performance/{args.var}_mse_results.csv')
        