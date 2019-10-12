#!/usr/bin/env python

'''
Floodfilling the contours.

Usage:
  floodfill.py [location to the images folder]

  Click on the image to set seed point

Keys:
  f     - toggle floating range
  c     - toggle 4/8 connectivity
  ESC   - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import imageio
from PIL import Image
def update(color=(0,0,0), dummy=None):
    if seed_pt is None:
        cv.imshow(cur_name_of_the_image, img)
        return
    mask[:] = 0
    lo = cv.getTrackbarPos('lo', cur_name_of_the_image)
    hi = cv.getTrackbarPos('hi', cur_name_of_the_image)
    flags = connectivity
    if fixed_range:
        flags |= cv.FLOODFILL_FIXED_RANGE

    if is_brush:
        cv.circle(img, seed_pt, 4, color, -1)
    else:
        cv.floodFill(img, mask, seed_pt, color, (lo,)*3, (hi,)*3, flags)

    cv.imshow(cur_name_of_the_image, img)

def onmouse(event, x, y, flags, param):
    global seed_pt
    if flags & cv.EVENT_FLAG_LBUTTON:
        seed_pt = x, y
        update(color)


if __name__ == '__main__':
    print(__doc__)

    FOLDER_TO_SAVE = '/home/bohdan/histologie/data/doctor_annotations/masks_corrections/unknown_correction/visible/'
    FOLDER_TO_IMGS = '/home/bohdan/histologie/data/doctor_annotations/masks_corrections/unknown_correction/visible/'
    img_names = os.listdir(FOLDER_TO_IMGS)
    for i in range(len(img_names)):

        cur_name_of_the_image =  img_names[i]
        img = cv.imread(os.path.join(FOLDER_TO_IMGS, cur_name_of_the_image))


        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        seed_pt = None
        fixed_range = True
        connectivity = 4
        cv.namedWindow(cur_name_of_the_image, cv.WINDOW_NORMAL)
        color = (0,0,0)
        is_brush = False

        update(color)
        cv.setMouseCallback(cur_name_of_the_image, onmouse)
        cv.createTrackbar('lo', cur_name_of_the_image, 20, 255, update)
        cv.createTrackbar('hi', cur_name_of_the_image, 20, 255, update)

        while True:
            ch = cv.waitKey()
            print(ch)
            if ch == 27:
                cv.destroyAllWindows()
                break
            if ch == ord('f'):
                fixed_range = not fixed_range
                print('using %s range' % ('floating', 'fixed')[fixed_range])
                update(color)
            if ch == ord('c'):
                connectivity = 12-connectivity
                print('connectivity =', connectivity)
                update(color)

            if ch == ord('0'):
                color = (0,0,0)
            if ch == ord('1'):
                color = (0,0,255)                
            if ch == ord('2'):
                color = (0,255,0)                
            if ch == ord('3'):
                color = (255,0,0)                
            if ch == ord('4'):
                color = (0,0,85)                
            if ch == ord('5'):
                color = (0,170,0)
            if ch == ord('6'):
                color = (127,0,255)
            if ch == ord('7'):
                color = (255,255,0)
            if ch == ord('8'):
                color = (0,85,0)
            if ch == ord('9'):
                color = (255,0,255)
            if ch == ord('p'):
                color = (0,85,255)
            if ch == ord('q'):
                color = (0,165,255)
            if ch == ord('w'):
                color = (0,255,255)
            if ch == ord('e'):
                color = (128,130,128)
            if ch == ord('b'):
                is_brush = not is_brush                
            if ch == 13:
                Image.fromarray(img[..., ::-1]).convert("RGB").save(os.path.join(FOLDER_TO_SAVE, cur_name_of_the_image))
                cv.destroyAllWindows()
                break
            print("Current color: ", color, "Current Instrument: Brush(",  is_brush, ')')
