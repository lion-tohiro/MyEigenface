from cv2 import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

train_sub_count = 40
train_img_count = 5
total_face = 200
row = 112
col = 92

def eigenfaces(src_path, a):
    img_list = np.empty((row*col, total_face))
    count = 0

    # read all the faces and flatten them
    for i in range(1, train_sub_count):
        for j in range(1, train_img_count):
            img_path = src_path + "/s" + str(i) + "/" + str(j) + ".pgm"
            # print(img_path)
            img = cv2.imread(img_path, 0)
            # cv2.imshow("1",img)
            img_col = np.array(img).flatten()

            img_list[:, count] = img_col[:]
            count += 1
    
    # compute the average of the faces
    img_mean = np.sum(img_list, axis=1) / total_face
    # average = img_mean.reshape(row, col).astype(np.uint8)
    # cv2.imshow("ave", average)
    # cv2.waitKey(0)

    # compute the difference matrix
    for i in range(0, total_face):
        img_list[:, i] -= img_mean[:]
    
    # compute the coveriance matrix
    cov = np.matrix(img_list.T)*np.matrix(img_list) / total_face

    # compute the eigen values and eigen vectors
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    sort_index = np.argsort(-eigen_values)
    eigen_values = eigen_values[sort_index]
    eigen_vectors = eigen_vectors[:, sort_index]
    # print(eigen_values)

    sum_eigen_values = sum(eigen_values)
    # print(sum_eigen_values)

    # according to a choose top k largest eigen values
    count_eigen_values = 0
    a_eigen_values = 0.0
    for value in eigen_values:
        count_eigen_values += 1
        a_eigen_values += value / sum_eigen_values
        if a_eigen_values >= a:
            break
    # print(count_eigen_values)
    eigen_values = eigen_values[0:count_eigen_values]
    eigen_vectors = eigen_vectors[:, 0:count_eigen_values]

    eigenfaces_mat = np.matrix(img_list)*np.matrix(eigen_vectors)

    eigenface_img = np.empty((row, col*10), dtype=np.uint8)
    for i in range(10):
        eigenface_img_temp = eigenfaces_mat.T[i].reshape(row, col).astype(np.uint8)
        # cv2.imshow(str(i), eigenface_img_temp)
        eigenface_img[:,col*i:col*(i+1)] = eigenface_img_temp
    cv2.imshow("eigenfaces",eigenface_img)
    cv2.waitKey(0)
    
    return img_mean, eigenfaces_mat

# the second parameter is always between 0.95-0.99
eigenfaces("D:\\k\\Myeigen\\MyEigenface\\att_faces", 0.95)