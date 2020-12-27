import os
import sys
from cv2 import cv2
import numpy as np
from pre_produce import pre_process

# some parameters of training and testing data
train_sub_count = 40
train_img_count = 5
total_face = 200
row = 112
col = 92
mask = [[25, 45], [66, 45]]

model_path = "D:\\k\\Myeigen\\MyEigenface\\model_1.npz"

def eigenfaces_detect(src, average, values, vectors, weight, a):
    src_img = cv2.imread(src+".pgm", 0)
    point = np.load(src+".npy")
    src_img_update = pre_process(src_img, point, mask)
    img_col = np.array(src_img_update).flatten()

    diff = img_col - average

    # according to a choose top k largest eigen values
    sum_eigen_values = sum(values)
    count_eigen_values = 0
    a_eigen_values = 0.0
    for value in values:
        count_eigen_values += 1
        a_eigen_values += value / sum_eigen_values
        if a_eigen_values >= a:
            break
    # print(count_eigen_values)
    vectors = vectors[:, 0:count_eigen_values]
    weight = weight[0:count_eigen_values, :]
    # print(weight)

    src_weight = np.matrix(vectors.T)*np.matrix(diff).T
    # print(src_weight)

    dis = []

    for i in range(total_face):
        distance = np.linalg.norm(weight.T[i]- src_weight.T)
        dis.append(distance)
    
    # get the index of detected face
    index = np.argsort(dis)
    print(index)

    # get the path of the detected face
    # get the subject dir
    sub = (int)(index[0] / 5 + 1)
    # print(sub)
    # get the number of the face in a certain subject
    number = index[0] % 5 + 1
    # print(number)
    # merge the subject and number to the path
    dist = "D:\\k\\Myeigen\\MyEigenface\\att_faces\\s" + str(sub) + "/" + str(number) + ".pgm"
    print("result: s{}\\{}.pgm".format(sub, number))
    dist_img = cv2.imread(dist, 0)

    detect = np.empty((row, col*2), dtype=np.uint8)
    detect[:, 0:col] = src_img
    detect[:, col:col*2] = dist_img
    cv2.imshow("detect", detect)

data = np.load(model_path)
average = data["average"]
values = data["values"]
vectors = data["vectors"]
weight = data["weight"]

eigenfaces_detect("D:\\k\\Myeigen\\MyEigenface\\att_faces\\s6\\6", average, values, vectors, weight, 0.95)

cv2.waitKey(0)