import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def get_color(img_name):
    img_ = img_name.replace('/','_')
    img = cv.imread(img_name)

    color = ('b', 'g', 'r')
    bins = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
    histograms = {}

    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256]).flatten()
        color_df = pd.DataFrame(columns=['intensity'], data=histr)
        aggregated_df = pd.DataFrame()
        for j in range(len(bins)-1):
            intensity_sum = color_df.loc[bins[j]:bins[j+1], 'intensity'].sum()
            aggregated_df.loc[j, 'intensity'] = intensity_sum
        histograms[col] = aggregated_df['intensity']
        # For sanity check
        # plt.plot(color_df, color=col) 

    #df has 3 columns for b,g,r and 16 rows 0-15
    df = pd.DataFrame(histograms)

    # Plot 3 histograms
    # for col in color:
    #     plt.figure()
    #     plt.bar(x = df.index ,height = df[col], color=col)
    #     plt.savefig(f'product/color3hist_{col}_{img_}')
    #     plt.close()

    # Plot heatmap
    # sns.heatmap(df, cmap='Greys')
    # plt.xlabel('Color')
    # plt.ylabel('Intensity Group')
    # plt.title('Color Histogram Heatmap')
    # plt.savefig(f'product/colorheat_{img_}')
    # plt.close()
    return df