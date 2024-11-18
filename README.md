# Deep-Learning-Microstructure-Property
MATLAB scripts for data preprocessing, optimization, training and interpretation of the deep learning framework for microstructure-composition-property relationships in Ni and Co based superalloys 

# Alloy dataset for training
The data used in this work has been collected from various literature which may contain copyrighted materials. Therefore we are unable to provide the data used for training. However, we have provided all the data sources in the file "Data_Sources.xlsx" located inside the "1_Alloy_Data" folder.

To replicate this work and use the training and interpretation scripts:

1. Collect images from the article sequentially from each data source and save as the serial number mentioned in the "Data_Sources.xlsx" using .PNG format. Place all the raw images inside a folder named "image_data".

2. Simultaneously collect the respective compositions and Vickers hardness and save in spreadsheet named "compositions.xls" in the following column order:"Image no.", "Al", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re", "C", "B", "Si", "P", "S", "Hardness"

3. Use the "batchprocess_images_in.m" function with 'aug' parameter value 0 for non-augmented dataset and value 1 for augmented dataset. The non-augmented image dataset containing resized images (224 x 224) is created with the name "images_original" and the augmented image dataset folder is created with the name "images_augmented".

4. Place all the data folders and files inside the 1_Alloy_Data folder. Please retain the organisation tree of all the files and folders, as most of the scripts use the relative file locations. If you wish to change, please modify the filepaths in the scripts appropriately before using.  

# Creating and training the modified DenseNet201 consisting of image inputs only
1. In the 2_DenseNet_201 folder the create_datastores function creates a combined datastore consisting of an image datastore and an array datastore. The combined datastore is then split into training and validation data using the set of indices specified in "data_split_for_tests.mat" file locataed in the "1_Alloy_Data" folder.

2. After creation of datastores, create the network layers using the create_d201 function. This function generates a layer graph object which can be used for training.

   (OR)

1. Use trainnetwork function to generate datastores, layers and fully trained network.  

