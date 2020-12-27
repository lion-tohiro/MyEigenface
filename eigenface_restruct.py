import os
import sys
from cv2 import cv2
import numpy as np
from base import normalize

# some parameters of training and testing data
train_sub_count = 40
train_img_count = 5
total_face = 200
row = 70
col = 70

def eigenfaces_restruct(src, average, values, vectors):
    src_img = cv2.imread(src+".png", 0)

    img_col = np.array(src_img).flatten()

    diff = img_col - average
    # print(diff)

    # 10PCs
    vectors_10pcs = vectors[:,0:10]
    # print(values)
    src_weight_10pcs = np.mat(vectors_10pcs.T)*np.mat(diff).T
    src_restruct_10pcs = np.mat(vectors_10pcs)*np.mat(src_weight_10pcs)
    src_restruct_10pcs = normalize(src_restruct_10pcs)
    result_10pcs = src_restruct_10pcs.T + average
    result_10pcs = normalize(result_10pcs)

    # 25PCs
    vectors_25pcs = vectors[:,0:25]
    src_weight_25pcs = np.mat(vectors_25pcs.T)*np.mat(diff).T
    src_restruct_25pcs = np.mat(vectors_25pcs)*np.mat(src_weight_25pcs)
    src_restruct_25pcs = normalize(src_restruct_25pcs)
    result_25pcs = src_restruct_25pcs.T + average
    result_25pcs = normalize(result_25pcs)

    # 50PCs
    vectors_50pcs = vectors[:,0:50]
    src_weight_50pcs = np.mat(vectors_50pcs.T)*np.mat(diff).T
    src_restruct_50pcs = np.mat(vectors_50pcs)*np.mat(src_weight_50pcs)
    src_restruct_50pcs = normalize(src_restruct_50pcs)
    result_50pcs = src_restruct_50pcs.T + average
    result_50pcs = normalize(result_50pcs)

    # 100PCs
    vectors_100pcs = vectors[:,0:100]
    src_weight_100pcs = np.mat(vectors_100pcs.T)*np.mat(diff).T
    src_restruct_100pcs = np.mat(vectors_100pcs)*np.mat(src_weight_100pcs)
    src_restruct_100pcs = normalize(src_restruct_100pcs)
    result_100pcs = src_restruct_100pcs.T + average
    result_100pcs = normalize(result_100pcs)

    # all
    src_weight = np.mat(vectors.T)*np.mat(diff).T
    src_restruct = np.mat(vectors)*np.mat(src_weight)
    src_restruct = normalize(src_restruct)
    result = src_restruct.T + average
    result = normalize(result)

    result_img = np.empty((row, col*6), dtype=np.uint8)
    result_img[:, 0:col] = src_img
    result_img[:, col:col*2] = re
    cv2.imshow("eigenfaces",eigenfaces_img)

    # result = result_10pcs.reshape(row, col).astype(np.uint8)
    # print(result)
    cv2.imshow("src", src_img)
    cv2.imshow("result_10pcs", result)