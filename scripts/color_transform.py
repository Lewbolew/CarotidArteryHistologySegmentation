import os
import numpy as np
import cv2 as cv
from PIL import Image
from tqdm import tqdm

def img_to_visible(img):
    visible_mask = np.zeros((img.shape[0], img.shape[1], 3), 'uint8')
    image = np.zeros((400,400,3), dtype="uint8")
    for i in range(len(mask_values)):
        visible_mask[np.where((img==mask_values[i]).all(axis=-1))] = real_colors[i]
    return visible_mask

def img_to_mask(img):
    original_mask = np.zeros((img.shape[0], img.shape[1], 3), 'uint8')
    image = np.zeros((400,400,3), dtype="uint8")
    for i in range(len(mask_values)):
        original_mask[np.where((img==real_colors[i]).all(axis=-1))] = mask_values[i]
    return original_mask


if __name__ == '__main__':

    FOLDER_WITH_IMGS_TO_TRANSFORM = '/home/bohdan/histologie/data/doctor_annotations/preprocessed/vis/unvisible/'
    FOLDER_TO_SAVE_TRANSFORMED_IMGS = '/home/bohdan/histologie/data/doctor_annotations/preprocessed/vis/visible/'
    TRANSFORM_MODE = 'to_colors' # can be two modes to_masks or to_colors

    # initialize mask values and corresponding color values. 
    mask_values = (
                    0,1,2,
                    3,4,5,
                    6,7,8,
                    9,10,11,
                    12,13
                   )
    # here not RGB but BGR because of OPENCV. 
    real_colors = (
                    (0,0,0), (0,0,255), (0,255,0),
                    (255,0,0), (0,0,85), (0,170,0),
                    (127,0,255), (255,255,0), (0,85,0),
                    (255,0,255), (0,85,255), (0,165,255),
                    (0,255,255), (128,130,128)
                   )
    if not os.path.exists(FOLDER_TO_SAVE_TRANSFORMED_IMGS):
        os.mkdir(FOLDER_TO_SAVE_TRANSFORMED_IMGS)

    # get names of all images in the folder
    img_names = os.listdir(FOLDER_WITH_IMGS_TO_TRANSFORM)

    for i in tqdm(range(len(img_names))):

        cur_img_name = img_names[i]
        cur_img = cv.imread(os.path.join(FOLDER_WITH_IMGS_TO_TRANSFORM, cur_img_name))
        if TRANSFORM_MODE == 'to_colors':
            converted_img = img_to_visible(cur_img)
        else:
            converted_img = img_to_mask(cur_img)
        convert_img_to_rgb = converted_img[..., ::-1]
        Image.fromarray(convert_img_to_rgb).convert("RGB").save(os.path.join(FOLDER_TO_SAVE_TRANSFORMED_IMGS, cur_img_name))
