import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import mahotas as mh
import os
import argparse
import mediapipe

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='rafdb_15k/DATASET/test/1') #rafdb_15k/DATASET/test
args = parser.parse_args()

img_dir = '/home/jiheeyou/subjective-output'

feat_df = pd.DataFrame()
fail_count = 0

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def create_feat_df(feat_df, img, img_name):
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    landmarks = results.multi_face_landmarks[0]
    face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
    df = pd.DataFrame(list(face_oval), columns = ["p1", "p2"])

    face_oval = mp_face_mesh.FACEMESH_FACE_OVAL

    df = pd.DataFrame(list(face_oval), columns = ["p1", "p2"])
    routes_idx = []
    
    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]
    
    for i in range(0, df.shape[0]):
        
        #print(p1, p2)
        
        obj = df[df["p1"] == p2]
        p1 = obj["p1"].values[0]
        p2 = obj["p2"].values[0]
        
        route_idx = []
        route_idx.append(p1)
        route_idx.append(p2)
        routes_idx.append(route_idx)

    routes = []

    #for source_idx, target_idx in mp_face_mesh.FACEMESH_FACE_OVAL:
    for source_idx, target_idx in routes_idx:
        
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]
            
        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

        #cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)
        
        routes.append(relative_source)
        routes.append(relative_target)

    mask = np.zeros((img.shape[0], img.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)
    out = np.zeros_like(img)
    out[mask] = img[mask]

    # fig = plt.figure(figsize = (7, 7))
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    features = {}
    for i in range(len(landmarks.landmark)):
        pt = landmarks.landmark[i]
        features[f'x{i}'] = pt.x # Between 0,1
        features[f'y{i}'] = pt.y # Between 0,1
        # plt.scatter(int(img.shape[1] * pt.x), int(img.shape[0] * pt.y), s=1, c='b')

    haralick_labels = ["Angular Second Moment",
                    "Contrast",
                    "Correlation",
                    "Sum of Squares: Variance",
                    "Inverse Difference Moment",
                    "Sum Average",
                    "Sum Variance",
                    "Sum Entropy",
                    "Entropy",
                    "Difference Variance",
                    "Difference Entropy",
                    "Information Measure of Correlation 1",
                    "Information Measure of Correlation 2",]
    haralick_features = mh.features.haralick(out,ignore_zeros=True,return_mean=True)

    for hl, hv in zip(haralick_labels, haralick_features):
        features[hl] = hv
    # plt.figure()
    # plt.bar(x=haralick_labels, height=haralick_features)
    # plt.xticks(rotation=90)

    mask = mask.astype("uint8")
    r_bins = cv2.calcHist([out],[0],mask,[32],[0,256])
    g_bins = cv2.calcHist([out],[1],mask,[32],[0,256])
    b_bins = cv2.calcHist([out],[2],mask,[32],[0,256])

    # plt.figure()
    # plt.plot(r_bins,color='red',label='Red Channel')
    # plt.plot(g_bins,color='green',label='Green Channel')
    # plt.plot(b_bins,color='blue',label='Blue Channel')
    # plt.xlabel('Intensity')
    # plt.ylabel('Pixels')
    # plt.legend()

    # plt.show()
    feat_row = pd.DataFrame(features,index=[0])
    feat_row['img_name'] = img_name.split('/')[-1]
    feat_row['label'] = img_name.split('/')[-2]
    feat_df = pd.concat([feat_df,feat_row])
    return feat_df

img_dir = os.path.join(img_dir, args.directory)
for nimg,file in enumerate([i for i in os.listdir(img_dir) if '.jpg' in i]):
    img = cv2.imread(os.path.join(img_dir,file))
    try: 
        feat_df = create_feat_df(feat_df, img, os.path.join(img_dir,file))
    except:
        print(f"Failed on {os.path.join(img_dir,file)}")
        fail_count += 1

path = 'product/'+os.path.join(img_dir, args.directory).replace('/','_')+'.csv'
feat_df.to_csv(path)
print(f"fail_count {fail_count}, success_count{len(feat_df)}")

# def coord_col(input_img):
#     coord_df = keypoint.get_keypoint(input_img)
#     num_coords = len(coord_df)
#     new_columns = []
#     for i in range(num_coords):
#         new_columns.append(f'x{i}')
#         new_columns.append(f'y{i}')
#     new_values = coord_df.values.flatten()
#     new_coord_df = pd.DataFrame([new_values], columns=new_columns)
#     return new_coord_df

# def texture_col(input_img):
#     texture_df = texture.get_texture(input_img).transpose()
#     texture_df.reset_index(drop=True, inplace=True)
#     texture_df.columns = texture_df.iloc[0]
#     new_texture_df = texture_df.drop(0).reset_index(drop=True)
#     return new_texture_df

# def color_col(input_img):
#     color_df = color.get_color(input_img)
#     new_columns = []
#     for i in range(16):
#         new_columns.append(f'b{i}')
#         new_columns.append(f'g{i}')
#         new_columns.append(f'r{i}')
#     new_values = color_df.values.flatten()
#     new_color_df = pd.DataFrame([new_values], columns=new_columns)
#     return new_color_df

# def one_row(input_img):
#     try:
#         new_color_df = color_col(input_img)
#     except:
#         columns = ['b0', 'g0', 'r0', 'b1', 'g1', 'r1', 'b2', 'g2', 'r2', 'b3', 'g3', 'r3',
#                 'b4', 'g4', 'r4', 'b5', 'g5', 'r5', 'b6', 'g6', 'r6', 'b7', 'g7', 'r7',
#                 'b8', 'g8', 'r8', 'b9', 'g9', 'r9', 'b10', 'g10', 'r10', 'b11', 'g11',
#                 'r11', 'b12', 'g12', 'r12', 'b13', 'g13', 'r13', 'b14', 'g14', 'r14',
#                 'b15', 'g15', 'r15']
#         values = np.full(len(columns), np.nan)
#         new_color_df = pd.DataFrame(columns=columns, data=[values])

#     try:
#         new_coord_df = coord_col(input_img)
#     except:
#         columns = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15', 'y15', 'x16', 'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20', 'y20', 'x21', 'y21', 'x22', 'y22', 'x23', 'y23', 'x24', 'y24', 'x25', 'y25', 'x26', 'y26', 'x27', 'y27', 'x28', 'y28', 'x29', 'y29', 'x30', 'y30', 'x31', 'y31', 'x32', 'y32', 'x33', 'y33', 'x34', 'y34', 'x35', 'y35', 'x36', 'y36', 'x37', 'y37', 'x38', 'y38', 'x39', 'y39', 'x40', 'y40', 'x41', 'y41', 'x42', 'y42', 'x43', 'y43', 'x44', 'y44', 'x45', 'y45', 'x46', 'y46', 'x47', 'y47', 'x48', 'y48', 'x49', 'y49', 'x50', 'y50', 'x51', 'y51', 'x52', 'y52', 'x53', 'y53', 'x54', 'y54', 'x55', 'y55', 'x56', 'y56', 'x57', 'y57', 'x58', 'y58', 'x59', 'y59', 'x60', 'y60', 'x61', 'y61', 'x62', 'y62', 'x63', 'y63', 'x64', 'y64', 'x65', 'y65', 'x66', 'y66', 'x67', 'y67']
#         values = np.full(len(columns), np.nan)
#         new_coord_df = pd.DataFrame(columns=columns, data=[values])
    
#     try:
#         new_texture_df = texture_col(input_img)
#     except:
#         columns = ['Angular Second Moment', 'Contrast', 'Correlation', 'Sum of Squares: Variance', 'Inverse Difference Moment', 'Sum Average', 'Sum Variance', 'Sum Entropy', 'Entropy', 'Difference Variance', 'Difference Entropy', 'Information Measure of Correlation 1', 'Information Measure of Correlation 2']
#         values = np.full(len(columns), np.nan)
#         new_texture_df = pd.DataFrame(columns=columns, data=[values])

#     row = new_color_df.join(new_coord_df).join(new_texture_df)
#     row.set_index(pd.Index([input_img]), inplace=True)
#     return row

# def folder_df(directory):
#     df = pd.DataFrame()
#     filenames = os.listdir(directory)
#     file_list = [os.path.join(directory, filename) for filename in filenames]
#     for file in file_list:
#         df = pd.concat([df, one_row(file)])
#     path = 'product/'+directory.replace('/','_')+'.csv'
#     df.to_csv(path)
#     return df

# folder_df(args.directory)

# #test_set
# #rafdb_15k/DATASET/test/1