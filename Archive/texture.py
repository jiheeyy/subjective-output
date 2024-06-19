import mahotas
import cv2 as cv
import numpy as np
from pylab import imshow, show
import pandas as pd
  
# loading image
def get_texture(img_name):
    img = mahotas.imread(img_name, as_grey=True).astype(int)
    
    # getting haralick features
    h_feature = mahotas.features.haralick(img)
    
    # sanity check
    # print("Haralick Features")
    # print(h_feature.shape)
    # imshow(h_feature)
    # show()

    h_features_uint8 = h_feature.astype(np.uint8)
    h_features_reshaped = np.reshape(np.transpose(h_features_uint8), (1, 13, 4))
    gray_image = cv.cvtColor(h_features_reshaped, cv.COLOR_RGBA2GRAY)[0]

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
                    #    "Maximal Correlation Coefficient"

    tex_dct = {label: val for label, val in zip(haralick_labels, gray_image)}
    texture_df = pd.DataFrame(data=tex_dct.items(), columns=['Haralick_Feature', 'Gray_Value'])
    return texture_df
