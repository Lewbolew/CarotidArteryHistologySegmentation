# HistologySegmentation
The segemntation of the Histological slices of the Carotid Artery

## Data description
  The clinician experts marked 166 images of histology of the carotid artery. There are
two types of histology images in our data with different stainings: light and dark.
Each slice of the carotid artery is represented with two types of staining. Correspondingly, 
we have 83 images of each type.  The dark type of images looks more feasible for semantic segmentation 
task as it contains much more color, texture and topology details about different classes. 
This dataset contains 12 classes. The quality of the annotations was very poor. The size of the input images 
varies a lot. We padded all images to the size of the image with maximum size with mirror padding and 
resized all images to 1024x1024 to be able to fit the GPU memory.

### Class color encodings
![histology_color_encoding](/imgs/histology_color_encoding.png)
