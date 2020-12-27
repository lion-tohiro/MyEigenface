import os
import sys
from cv2 import cv2
import numpy as np

row = 112
col = 92

def pre_process(img, point, mask):
    img_temp = img.copy()

    eye_vector = [point[1][0]-point[0][0], point[1][1]-point[0][1]]
    mask_vector = [mask[1][0]-mask[0][0], mask[1][1]-mask[0][1]]
    eye_mask = [mask[0][0]-point[0][0], mask[0][1]-point[0][1]]

    # scale
    dis_eye = np.linalg.norm(eye_vector)
    dis_mask = np.linalg.norm(mask_vector)
    scale = dis_mask / dis_eye

    # rotate
    tan = eye_vector[1] / eye_vector[0]
    theta = np.arctan(tan) * 180 / np.pi

    # transform
    transform_x = eye_mask[0]
    transform_y = eye_mask[1]

    # do rotate and scale
    rotate_scale_mat = cv2.getRotationMatrix2D((point[0][0], point[0][1]), theta, scale)
    img_temp = cv2.warpAffine(img_temp, rotate_scale_mat, (col, row))

    # do transform
    transform_mat = np.float32([[1, 0, transform_x], [0, 1, transform_y]])
    img_temp = cv2.warpAffine(img_temp, transform_mat, (col, row))

    # cv2.circle(img_temp, (mask[0][0], mask[0][1]), 2, (255,0,0), -1)

    img_update = img_temp.copy()

    # do histogram equalization
    img_update = cv2.equalizeHist(img_update)

    return img_update

# src = "D:\\k\\Myeigen\\MyEigenface\\att_faces\\s35\\2.pgm"
# img = cv2.imread(src)
# mask = [[25, 45], [66, 45]]
# src_p = "D:\\k\\Myeigen\\MyEigenface\\att_faces\\s35\\2.npy"
# point = np.load(src_p)

# img_update = pre_process(img, point, mask)

# cv2.imshow("src", img)
# cv2.imshow("update", img_update)

# cv2.waitKey(0)