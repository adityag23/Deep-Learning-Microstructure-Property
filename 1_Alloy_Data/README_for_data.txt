The data used in this work has been collected from various literature which may contain copyrighted materials.

Therefore we are unable to provide the data used for training. However, we have provided all the data sources in the file "Data_Sources.xlsx"

To replicate this work and use the training and interpretation scripts:

1. Collect images from the article sequentially from each data source and save as the serial number mentioned in the "Data_Sources.xlsx" using .PNG format. Place all the raw images inside a folder named "image_data".

2. Simultaneously collect the respective compositions and Vickers hardness and save in spreadsheet named "compositions.xls" in the following column order:"Image no.", "Al", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re", "C", "B", "Si", "P", "S", "Hardness"

3. Use the "batchprocess_images_in.m" function with 'aug' parameter value 0 for non-augmented dataset and value 1 for augmented dataset.

The dataset is ready for use.
