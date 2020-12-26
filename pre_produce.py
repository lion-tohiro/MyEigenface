import os
import sys
from cv2 import cv2
import numpy as np

def pre_process(img, point, mask):
    img_update = img.copy()

    eye_vector = [point[1][0]-point[0][0], point[1][1]-point[0][1]]
    mask_vector = [mask[1][0]-mask[0][0], mask[1][1]-mask[0][1]]
    eye_mask = [mask[0][0]-point[0][0], mask[0][1]-point[0][1]]

    tan = eye_vector[1] / eye_vector[0]


    return img_update