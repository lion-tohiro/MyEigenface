from cv2 import cv2
import numpy as np
import sys
import os
import PIL

train_sub_count = 40
train_img_count = 5

def read_image(src_path):
    img_list = []

    for i in range(1, train_sub_count + 1):
        for j in range(1, 6):
            img_path = src_path + "/s" + str(i) + "/" + str(j) + ".pgm"
            img = cv2.imread(img_path)
            img_list.append(img)
    
    return img_list

