import numpy as np
import pandas as pd
import mahotas
from pylab import imshow, show
import keypoint
import texture
import color
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='test_set')
args = parser.parse_args()

def coord_col(input_img):
    coord_df = keypoint.get_keypoint(input_img)
    num_coords = len(coord_df)
    new_columns = []
    for i in range(num_coords):
        new_columns.append(f'x{i}')
        new_columns.append(f'y{i}')
    new_values = coord_df.values.flatten()
    new_coord_df = pd.DataFrame([new_values], columns=new_columns)
    return new_coord_df

def texture_col(input_img):
    texture_df = texture.get_texture(input_img).transpose()
    texture_df.reset_index(drop=True, inplace=True)
    texture_df.columns = texture_df.iloc[0]
    new_texture_df = texture_df.drop(0).reset_index(drop=True)
    return new_texture_df

def color_col(input_img):
    color_df = color.get_color(input_img)
    new_columns = []
    for i in range(16):
        new_columns.append(f'b{i}')
        new_columns.append(f'g{i}')
        new_columns.append(f'r{i}')
    new_values = color_df.values.flatten()
    new_color_df = pd.DataFrame([new_values], columns=new_columns)
    return new_color_df

def one_row(input_img):
    try:
        new_color_df = color_col(input_img)
    except:
        columns = ['b0', 'g0', 'r0', 'b1', 'g1', 'r1', 'b2', 'g2', 'r2', 'b3', 'g3', 'r3',
                'b4', 'g4', 'r4', 'b5', 'g5', 'r5', 'b6', 'g6', 'r6', 'b7', 'g7', 'r7',
                'b8', 'g8', 'r8', 'b9', 'g9', 'r9', 'b10', 'g10', 'r10', 'b11', 'g11',
                'r11', 'b12', 'g12', 'r12', 'b13', 'g13', 'r13', 'b14', 'g14', 'r14',
                'b15', 'g15', 'r15']
        values = np.full(len(columns), np.nan)
        new_color_df = pd.DataFrame(columns=columns, data=[values])

    try:
        new_coord_df = coord_col(input_img)
    except:
        columns = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15', 'y15', 'x16', 'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20', 'y20', 'x21', 'y21', 'x22', 'y22', 'x23', 'y23', 'x24', 'y24', 'x25', 'y25', 'x26', 'y26', 'x27', 'y27', 'x28', 'y28', 'x29', 'y29', 'x30', 'y30', 'x31', 'y31', 'x32', 'y32', 'x33', 'y33', 'x34', 'y34', 'x35', 'y35', 'x36', 'y36', 'x37', 'y37', 'x38', 'y38', 'x39', 'y39', 'x40', 'y40', 'x41', 'y41', 'x42', 'y42', 'x43', 'y43', 'x44', 'y44', 'x45', 'y45', 'x46', 'y46', 'x47', 'y47', 'x48', 'y48', 'x49', 'y49', 'x50', 'y50', 'x51', 'y51', 'x52', 'y52', 'x53', 'y53', 'x54', 'y54', 'x55', 'y55', 'x56', 'y56', 'x57', 'y57', 'x58', 'y58', 'x59', 'y59', 'x60', 'y60', 'x61', 'y61', 'x62', 'y62', 'x63', 'y63', 'x64', 'y64', 'x65', 'y65', 'x66', 'y66', 'x67', 'y67']
        values = np.full(len(columns), np.nan)
        new_coord_df = pd.DataFrame(columns=columns, data=[values])
    
    try:
        new_texture_df = texture_col(input_img)
    except:
        columns = ['Angular Second Moment', 'Contrast', 'Correlation', 'Sum of Squares: Variance', 'Inverse Difference Moment', 'Sum Average', 'Sum Variance', 'Sum Entropy', 'Entropy', 'Difference Variance', 'Difference Entropy', 'Information Measure of Correlation 1', 'Information Measure of Correlation 2']
        values = np.full(len(columns), np.nan)
        new_texture_df = pd.DataFrame(columns=columns, data=[values])

    row = new_color_df.join(new_coord_df).join(new_texture_df)
    row.set_index(pd.Index([input_img]), inplace=True)
    return row

def folder_df(directory):
    df = pd.DataFrame()
    filenames = os.listdir(directory)
    file_list = [os.path.join(directory, filename) for filename in filenames]
    for file in file_list:
        df = pd.concat([df, one_row(file)])
    path = 'product/'+directory.replace('/','_')+'.csv'
    df.to_csv(path)
    return df

folder_df(args.directory)

#test_set
#rafdb_15k/DATASET/test/1