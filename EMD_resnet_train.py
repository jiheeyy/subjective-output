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

# Seed and Device Initialization
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15, help="Number of epochs",)
parser.add_argument("--attribute", type=str, default='trustworthy')
parser.add_argument("--transform", type=str, default="none")
parser.add_argument("--var", type=str, default="none") # all50, frozen+regressor, transfer, initial
args = parser.parse_args()

num_epochs = args.epochs
var = args.var
column = args.attribute
transform = args.transform

truemean = pd.read_csv('attribute_means.csv')

def define_model(var):
    if var == 'initial':
        model = resnet50(weights=None)
    else:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) # V1 matches tensorflow weights = 'imagenet'

    # Freeze all layers except the final classifier
    if var == "all50" or var == "frozen+regressor": # TrashTrue, TrashFalse
        for param in model.parameters():
            param.requires_grad = False
    # Unfreeze
    elif var == "transfer" or var == "initial":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Invalid var: {var}")

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

    # Unfreeze only the newly added layers
    for param in model.fc.parameters():
        param.requires_grad = True

    # Verify which layers are trainable
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    return model

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

# Create train dataloader
train_dataset = OMI_Dataset('train_set')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create all50 train dataloader
# Input=50 since rating ranges from 0 to 100
all50_train_dataset = OMI_Dataset('train_set', var='all50')
all50_train_dataloader = DataLoader(all50_train_dataset, batch_size=32, shuffle=True)

# Create test dataloader
test_dataset = OMI_Dataset('test_set')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Testing 
def evaluate_model(model, test_dataloader):
    model.to(device)
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_images, batch_labels in test_dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_images).squeeze(1)
            
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_dataloader)
    print(f"Testing Loss: {avg_loss:.4f}")
    return avg_loss

# Training loop
def train_model(model, train_dataloader, num_epochs=10):
    model.to(device)
    print("device: ", device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_images, batch_labels in train_dataloader:
            # Move data to device
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_images).squeeze(1)
            
            # Compute loss
            loss = criterion(outputs, batch_labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
        
        # Compute average loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        
        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        if epoch % 10 == 0:
            evaluate_model(model, test_dataloader)
        
        # Step the scheduler
        scheduler.step(avg_loss)
    
    return model

# Train the model.
if var == 'all50':
    data_loader = all50_train_dataloader
else:
    data_loader = train_dataloader

model = define_model(var)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Learning rate scheduler (optional, but helpful)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

if var != 'initial': 
    trained_model = train_model(model, data_loader, num_epochs=num_epochs)
elif var == 'initial':
    trained_model = model

# Save the trained model
save_path = f'weights/{var}/trained_epoch{num_epochs}_attribute{column}_transform{transform}_weights.pth'
torch.save(trained_model.state_dict(), save_path)
print(f"Model saved.", save_path)

print("Final Evaluation")
evaluate_model(trained_model, test_dataloader)


# Additional Metrics
# Lists to store predictions and true labels
all_predictions = []
all_labels = []

# Disable gradient computation for inference
with torch.no_grad():
    for batch_images, batch_labels in test_dataloader:
        # Move data to device
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        outputs = trained_model(batch_images)
        
        # Convert to numpy for metric calculation
        predictions = outputs.cpu().numpy()
        labels = batch_labels.cpu().numpy()
        
        # Append to lists
        all_predictions.extend(predictions)
        all_labels.extend(labels)

# Converts list of arrays into list
# To match all_labels
all_predictions = [item[0] for item in all_predictions]

#  Calculate performance metrics
mse = mean_squared_error(all_labels, all_predictions)
mae = mean_absolute_error(all_labels, all_predictions)
rmse = np.sqrt(mse)
correlation = np.corrcoef(all_labels, all_predictions)[0, 1]

# Print results
print("Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Correlation Coefficient: {correlation:.4f}")