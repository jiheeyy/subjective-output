from scipy.stats import wasserstein_distance, wasserstein_distance_nd
import os
import cv2
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import shap
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
import time
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import argparse
from ot.sliced import sliced_wasserstein_distance as sliced_wasserstein

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="init_transfer", help="Mode for the script")
args = parser.parse_args()

mode = args.mode

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

truemean = pd.read_csv('attribute_means.csv')
def true_mean(image_name, column):
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
    return truemean[column][stim - 1]  # For i+1.jpg, should look for truemean['happy'][i]
    
def test_lasso_mse(train_x, train_y, test_x, test_y):
    scaler = StandardScaler()
        
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    
    alpha_value = 0.1
    lasso = Lasso(alpha=alpha_value, max_iter=4000)
    lasso.fit(train_x_scaled, train_y)

    train_y_pred = lasso.predict(train_x_scaled)
    test_y_pred = lasso.predict(test_x_scaled)

    test_mse = mean_squared_error(test_y, test_y_pred)
    return lasso.coef_, test_mse

def mc_pairwise_distance(coords, n_samples=100000):
    N = coords.shape[0]
    idx1 = np.random.randint(0, N, n_samples)
    idx2 = np.random.randint(0, N, n_samples)
    
    for i in range(len(idx1)):
        if idx1[i] == idx2[i]:
            idx2[i] = np.random.choice(np.delete(np.arange(N), idx1[i]))
    
    distances = np.linalg.norm(coords[idx1] - coords[idx2], axis=1)
    return np.mean(distances)

def preprocess_array(array):
    array = array.sum(axis=2).squeeze() # sum along color channels
    array += 1 # elevate piles of dirt
    array /= array.sum() # treat weights as probability distribution, rather than dirt piles

    coords = np.indices(array.shape).reshape(2,-1).T
    values = array.ravel()
    return coords, values

def shap_emd(array1, array2):
    # 1 dummy, 2 real
    start_time = time.time()
    coords1, values1 = preprocess_array(array1)
    coords2, values2 = preprocess_array(array2)
    emd = sliced_wasserstein(X_s = coords1, X_t = coords2, a = values1, b = values2, seed=0)
    avg_dist = mc_pairwise_distance(coords1)
    print("Time to compute shap EMD:", time.time() - start_time)
    print("SHAP EMD:", emd, "Avg Dist:", avg_dist)
    return emd / avg_dist

def lasso_emd(weights):
    #1 real, 2 dummy
    start_time = time.time()
    values1 = weights.reshape(-1, 1)
    values1 += 1 # elevate piles of dirt
    values1 /= values1.sum() # treat weights as probability distribution, rather than dirt piles
    values2 = np.full(values1.shape, values1.mean())
    coords = np.eye(len(values1))
    emd = sliced_wasserstein(X_s = coords, X_t = coords, a = values1, b = values2, seed=0)
    avg_dist = mc_pairwise_distance(coords)
    print("Time to compute lasso EMD:", time.time() - start_time)
    print("Lasso EMD:", emd, "Avg Dist:", avg_dist)
    return emd / avg_dist


trashfalse_df = pd.read_csv('performance/trashfalse_mse_results.csv')
initial_df = pd.read_csv('performance/initial_mse_results.csv')
transfer_df = pd.read_csv('performance/transfer_mse_results.csv')

# Load the SHAP values across all 34 attributes
finished_attributes = [
    "trustworthy", "attractive", "dominant", "smart", "age", "gender", "weight", 
    "typical", "happy", "familiar", "outgoing", "memorable", "well-groomed", 
    "long-haired", "smug", "dorky", "skin-color", "hair-color", "alert", "cute", 
    "privileged", "liberal", "asian", "middle-eastern", "hispanic", "islander", 
    "native", "black", "white", "looks-like-you", "gay", "electable", "godly", 
    "outdoors"
]

attributes = []
emd_values0 = []
emd_values1 = []
lasso_emd_values = []
lasso_acc_values = []
shap_metrics = pd.DataFrame()
lasso_metrics = []

for fa in finished_attributes:
    if mode =="trash_nottrash": #nottrash is resnet weights frozen + trainable linear layer
        directory1 = f'shap_dict/trash2/{fa}_trashTrue_nimg2.pkl' # "dir1 corresponds to weird weights"
        directory2 = f'shap_dict/trash2/{fa}_trashFalse_nimg2.pkl'
    elif mode == "init_transfer":
        directory1 = f'shap_dict/initial/{fa}_nimg2.pkl'
        directory2 = f'shap_dict/transfer/{fa}_nimg2.pkl'
    elif mode == "init_nottrash":
        directory1 = f'shap_dict/initial/{fa}_nimg2.pkl'
        directory2 = f'shap_dict/trash2/{fa}_trashFalse_nimg2.pkl'
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    with open(directory1, 'rb') as f:
        shap_values1 = pickle.load(f)
    with open(directory2, 'rb') as f:
        shap_values2 = pickle.load(f)
    
    flattened_shapvalue_img0_model1 = np.array(shap_values1[0].values).flatten()
    flattened_shapvalue_img1_model1 = np.array(shap_values1[1].values).flatten()
    flattened_shapvalue_img0_model2 = np.array(shap_values2[0].values).flatten()
    flattened_shapvalue_img1_model2 = np.array(shap_values2[1].values).flatten()
    spearman_img0 = stats.spearmanr(flattened_shapvalue_img0_model1, flattened_shapvalue_img0_model2)
    spearman_img1 = stats.spearmanr(flattened_shapvalue_img1_model1, flattened_shapvalue_img1_model2)
    pearson_img0 = stats.pearsonr(flattened_shapvalue_img0_model1, flattened_shapvalue_img0_model2)
    pearson_img1 = stats.pearsonr(flattened_shapvalue_img1_model1, flattened_shapvalue_img1_model2)
    euc_img0 = np.linalg.norm(flattened_shapvalue_img0_model1 - flattened_shapvalue_img0_model2)
    euc_img1 = np.linalg.norm(flattened_shapvalue_img1_model1 - flattened_shapvalue_img1_model2)

    plt.figure()
    
    shap_metric = np.array([spearman_img0.statistic, spearman_img1.statistic, pearson_img0.statistic, pearson_img1.statistic, 
    spearman_img0.pvalue, spearman_img1.pvalue, pearson_img0.pvalue, pearson_img1.pvalue, euc_img0, euc_img1])

    fa_emd0 = shap_emd(shap_values1[0].values,
                     shap_values2[0].values) # Model 1, Model 2 weights on the same image 0
    fa_emd1 = shap_emd(shap_values1[1].values,
                     shap_values2[1].values)


    attributes.append(fa)
    emd_values0.append(fa_emd0)
    emd_values1.append(fa_emd1)
    shap_metric = shap_metric.reshape(1, -1)
    if shap_metrics.empty:
        shap_metrics = pd.DataFrame(shap_metric)
    else:
        shap_metrics = pd.concat([shap_metrics, pd.DataFrame(shap_metric)])

    train_x = pd.read_csv('alfred/alfred_train.csv')
    test_x = pd.read_csv('alfred/alfred_test.csv')
    train_x['stim_id'] = train_x['stim_id'].apply(lambda x: true_mean(x, fa))
    test_x['stim_id'] = test_x['stim_id'].apply(lambda x: true_mean(x, fa))
    lasso_coef, lasso_acc = test_lasso_mse(train_x.drop(columns=['stim_id']), train_x['stim_id'], test_x.drop(columns=['stim_id']), test_x['stim_id'])
    lasso_emd_values.append(lasso_emd(lasso_coef))
    lasso_acc_values.append(lasso_acc)

    flat_coef = np.array(lasso_coef).flatten()
    flat_meancoef = np.full(flat_coef.shape, lasso_coef.mean())
    # spearman_lasso = stats.spearmanr(flat_coef, flat_meancoef)
    euc_lasso = np.linalg.norm(flat_coef - flat_meancoef)
    lasso_metric = euc_lasso
    lasso_metrics.append(lasso_metric)

# Create DataFrame
emd_df = pd.DataFrame({
    'Attribute': attributes,
    'EMD Img 0': emd_values0,
    'EMD Img 1': emd_values1,
    'Lasso_EMD': lasso_emd_values,
    'Lasso MSE': lasso_acc_values,
    'Shap Spearman Img 0': shap_metrics.iloc[:,0],
    'Shap Spearman Img 1': shap_metrics.iloc[:,1],
    'Shap Pearson Img 0': shap_metrics.iloc[:,2],
    'Shap Pearson Img 1': shap_metrics.iloc[:,3],
    'Shap Spearman Pvalue Img 0': shap_metrics.iloc[:,4],
    'Shap Spearman Pvalue Img 1': shap_metrics.iloc[:,5],
    'Shap Pearson Pvalue Img 0': shap_metrics.iloc[:,6],
    'Shap Pearson Pvalue Img 1': shap_metrics.iloc[:,7],
    'Shap Euclidean Img 0': shap_metrics.iloc[:,8],
    'Shap Euclidean Img 1': shap_metrics.iloc[:,9],
    'Lasso Euclidean': lasso_metrics,
})

# if mode == 'init_nottrash':
#     final_df = pd.concat([emd_df, trashfalse_df[['average_mse']]], axis=1)
#     final_df = final_df.rename(columns={'average_mse': "Frozen Resnet + Linear Avg MSE"})
#     final_df = pd.concat([final_df, trashfalse_df[['average_mse']] / initial_df[['average_mse']] * 100], axis=1)
# elif mode == 'init_transfer':
#     final_df = pd.concat([emd_df, transfer_df[['average_mse']]], axis=1)
#     final_df = final_df.rename(columns={'average_mse': "Transfer Avg MSE"})
#     final_df = pd.concat([final_df, transfer_df[['average_mse']] / initial_df[['average_mse']] * 100], axis=1)
# else:
#     print("final_df not created")
# final_df = final_df.rename(columns={'average_mse': "All Test Images - MSE Ratio in %"})
final_df = emd_df.copy()
final_df.loc[:, final_df.columns != "attribute"] = final_df.loc[:, final_df.columns != "attribute"].round(5)
final_df.to_csv(f'performance/test/marc_{mode}.csv', index=False)