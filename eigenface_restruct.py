import os
import sys
from cv2 import cv2
import numpy as np

# some parameters of training and testing data
train_sub_count = 40
train_img_count = 5
total_face = 200
row = 112
col = 92

model_path = "D:\\k\\Myeigen\\MyEigenface\\model.npz"

def eigenfaces_restruct(src, average, values, vectors):
    src_img = cv2.imread(src, 0)
    print(src_img)
    cv2.imshow("src", src_img)
    img_col = np.array(src_img).flatten()

    diff = img_col - average
    # print(diff)

    # 10PCs
    vectors_10pcs = vectors[:, 0:10]
    src_weight_10pcs = np.matrix(vectors_10pcs.T)*np.matrix(diff).T
    src_restruct_10pcs = np.matrix(vectors_10pcs)*np.matrix(src_weight_10pcs)
    src_restruct_10pcs = 255*(src_restruct_10pcs - np.min(src_restruct_10pcs)) / (np.max(src_restruct_10pcs) - np.min(src_restruct_10pcs))
    result_10pcs = src_restruct_10pcs.T+average
    # print(np.size(result_10pcs,0))
    result_10pcs = 255*(result_10pcs - np.min(result_10pcs)) / (np.max(result_10pcs) - np.min(result_10pcs))
    result = result_10pcs.reshape(row, col).astype(np.uint8)
    print(result)
    cv2.imshow("result_10pcs", result)

data = np.load(model_path)
average = data["average"]
values = data["values"]
vectors = data["vectors"]
# weight = data["weight"]

eigenfaces_restruct("D:\\k\\Myeigen\\MyEigenface\\att_faces\\s2\\7.pgm", average, values, vectors)

cv2.waitKey(0)