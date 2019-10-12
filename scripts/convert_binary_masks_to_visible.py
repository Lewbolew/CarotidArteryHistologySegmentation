import os
import numpy as np
import cv2 as cv
from PIL import Image
from tqdm import tqdm
def mask_to_visible(img):
    img[img==1] = 255
    img[img==2] = 0
    return img


if __name__ == '__main__':

    PATH_TO_MASKS = '/home/bohdan/histologie/data/doctor_annotations/export_results/'
    PATH_TO_SAVE = '/home/bohdan/histologie/data/doctor_annotations/visible_binary_masks/'

    if not os.path.exists(PATH_TO_SAVE):
        os.mkdir(PATH_TO_SAVE)
    mask_names = os.listdir(PATH_TO_MASKS)

    for i in tqdm(range(len(mask_names))):
        cur_mask = cv.imread(os.path.join(PATH_TO_MASKS, mask_names[i]))
        visible_mask = mask_to_visible(cur_mask)
        Image.fromarray(visible_mask).convert("RGB").save(os.path.join(PATH_TO_SAVE, mask_names[i]))
