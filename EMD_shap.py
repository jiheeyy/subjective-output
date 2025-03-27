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
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import time
import pickle

argparser = argparse.ArgumentParser()
argparser.add_argument('--column', type=str, default='happy')
argparser.add_argument('--transform', type=str, default='resnet')
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--max_evals', type=int, default=50176) # 224 * 224
argparser.add_argument('--nimg', type=int, default=1) # Number of images to evaluate
argparser.add_argument('--var', type=str, default="none", help="all50, frozen+regressor, transfer, initial")
args = argparser.parse_args()

column = args.column
transform = args.transform
seed = args.seed #
max_evals = args.max_evals
nimg = args.nimg #
var = args.var

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

start_time = time.time()

# Prediction function that scales output
def f(x):
    with torch.no_grad():
        tmp = x.copy()
        tmp = np.transpose(tmp, (0, 3, 1, 2)) # Resnet requires (N, C, H, W)
        tmp = torch.from_numpy(tmp).float()
        output = loaded_model(tmp)
        return output

test_dataset = OMI_Dataset('test_set', transform="resnet")
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
for test_images, test_labels in test_dataloader:
    test_images = test_images.permute(0, 2, 3, 1) # SHAP expects (N, H, W, C)

    # Create a blur masker
    masker_blur = shap.maskers.Image("blur(128,128)", test_images[0].shape)

    # Create SHAP explainer
    explainer = shap.Explainer(f, masker_blur)

    # Explain a subset of the test data
    # SHAP expects numpy array
    images_to_explain = test_images[:nimg].numpy()
    labels_to_explain = test_labels[:nimg]

    # Compute SHAP values
    shap_values = explainer(images_to_explain, max_evals=max_evals, batch_size=8)
    print("Time to compute SHAP values:", time.time() - start_time)
    break # since I'm calculating for only 2 images right now

# Save the SHAP values
if var == 'transfer' or var == 'initial':
    with open(f'shap_dict/{var}/{column}_nimg{nimg}.pkl', 'wb') as f:
        pickle.dump(shap_values, f)
    for i in range(nimg):
        shap.image_plot(shap_values[i])
        plt.savefig(f'shap_dict/{var}/{column}_whichimg{i}.png')

else:
    with open(f'shap_dict/{var}/{column}_nimg{nimg}.pkl', 'wb') as f:
        pickle.dump(shap_values, f)
    for i in range(nimg):
        shap.image_plot(shap_values[i])
        plt.savefig(f'shap_dict/{var}/{column}_whichimg{i}.png')

# Load the SHAP values
# with open(f'shap_dict/{column}_seed{seed}_nimg{nimg}.pkl', 'rb') as f:
#     loaded_shap_values = pickle.load(f)