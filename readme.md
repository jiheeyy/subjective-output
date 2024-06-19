# Applying Modular Model to RAF-DB Dataset


## These five files should be the most relevant:
* disc.yml
    Conda environment package specifications for running the below files.
* main.py
    For all face images in the folder you specify, this python script generates a single dataframe with color bins, haralick texture, and openface keypoint information. Each row of the dataframe contains information about a single image.
* main.sbatch
    If you want to run main.py on a computing cluster, login to cluster, then run sbatch main.sbatch. You may want to reference submit.sbatch in the Archive as well.
* Extracting Facial Features Using DNNs - Amazon and Openface.ipynb
    This jupyter notebook illustrates what facial feature extraction with Amazon and Openface each looks like. Note that Amazon requires public/private key setup, and Openface requires docker activity.
* Testing Color, Texture, Keypoint Extraction Pipeline with Face Masking.ipynb
    This notebook illustrates processes in main.py, therefore is useful for sanity check on main.py.
* Visualizing Categorical Classification, MSE Regression Weights with Altair.ipynb
    This notebook contains examples on logistic regression on both categorical problem and regression problem. Also, it contains attempts to visualize modular model weights interactively using altair.
    
    
## These directories can be useful:
* logs 
    Contains .err and .out files from cluster batch submissions
* rafdb_15k
    Contains publicly available version of RAF-DB dataset. While the original RAF-DB dataset contained about 30k images, POSTERV2 used the 15k version instead for benchmarking. Therefore, I downloaded the 15k version from Kaggle to use for our project as well.
* rafdb_product
    Contains products from main.py on RAF-DB dataset. You are good to create your own using main.sbatch and main.py, but you are also free to use the dataframes I've already created.
* test_set
    Contains test set images from One Million Impressions dataset (the same ~400 images Alfred uses for testing)
* train_set
    Contains train set images from One Million Impressions dataset (the same ~600 images Alfred uses for training)
* Archive
    A fun place. It also contains previous dataframes based on openface and amazon for OMI dataset. Files in this folder are not immediately relevant for applying modular model to RAF-DB dataset, but may be useful.

## TODO
When you extract color bins, haralick texture, and openface keypoint information from RAF-DB dataset, you will run into quite a lot of openface detection failure cases. Your role is to investigate why so many failures may be happening, then reduce failure cases so that you can get a full accessment on how well the modular model performs on RAF-DB dataset.

Some ideas are:
- Sharpen RAF-DB dataset using image kernels, see if that reduces failure cases.
- Average pixels for failure cases, then average pixels for success cases. See if you see a notable difference. You could also look at where the pixels over one SD are distributed; and where those over two SD are distributed.
- Find Shannon entropy value for each of the red, blue, green channels. Compare entropy values for failure cases vs. success cases.
- With image kernel, extract edge coordinates for each image. Then stack edges from ~100 failure images, and separately stack edges from ~100 success images. See if you can detect a different edge pattern in failure images.
- You could try running SHAP, a black box explainability model, to see if failure cases exhibit a different SHAP weight pattern.

This is not directly related to reducing failure cases, but it would be nice to have this additional facial feature information:
- Consider calculating 3d angle of the face using the face mesh, and add that information to the dataframes.

