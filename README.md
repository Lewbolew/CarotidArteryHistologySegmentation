# HistologySegmentation
The segemntation of the Histological slices of the Carotid Artery

## Method
This github contains the implementation of the "Learning and Incorporating Shape Models for Semantic Segmentation" paper:

https://www.researchgate.net/publication/314256462_Learning_and_Incorporating_Shape_Models_for_Semantic_Segmentation

![shape_framework](/imgs/shape_framework.png)

## Shape prior model
![SmallShapeNetArchitecture](/imgs/SmallShapeNetArchitecture.png)

## Segmenation models
- U-Net
- Attention R2U-Net
- TernausNetV2

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

### Data example
 <img src="/imgs/x.png" width="250" height="250" alt="Optional title"> <img src="/imgs/y.png" width="250" height="250">
 
### Prediction Examples
![histology_prediction](/imgs/histology_prediction.png)
