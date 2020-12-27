from cv2 import cv2
import numpy as np
import sys
import os
from base import normalize

# some parameters of training and testing data
train_sub_count = 40
train_img_count = 5
total_face = 200
row = 70
col = 70

def eigenfaces_train(src_path):
    img_list = np.empty((row*col, total_face))
    count = 0

    # read all the faces and flatten them
    for i in range(1, train_sub_count+1):
        for j in range(1, train_img_count+1):
            img_path = src_path + "/s" + str(i) + "/" + str(j) + ".png"
            img = cv2.imread(img_path, 0)

            img_col = np.array(img).flatten()
            img_list[:, count] = img_col[:]
            count += 1
    
    # compute the average of the faces
    img_mean = np.sum(img_list, axis=1) / total_face

    diff = np.empty((row*col, total_face))
    # compute the difference matrix
    for i in range(0, total_face):
        diff[:, i] = img_list[:, i] - img_mean[:]

    cov = np.mat(diff)*np.mat(diff.T) / total_face
    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    # sort the eigenvalues and eigenvectors by desc
    sort_index = np.argsort(-eigen_values)
    eigen_values = eigen_values[sort_index]
    eigen_vectors = eigen_vectors[:, sort_index]
    # print(eigen_values)
    '''
    compute the coveriance matrix
    here we don't use original algrithom to avoid computing an 10000+ * 10000+ coveriance matrix later
    oringinal: cov = 1/m * A*A^T => it will be an 10000+ * 10000+ matrix
    when the dimension of the image (here we mean row*col) > the total number of the training images (here we mean total_face)
    (1)cov*v = A*A^T*v = e*v (e is eigenvalue of cov, v is eigenvector of cov) => original
    (2)let cov'*u = A^T*A*u = e*u
    thus, on both sides of the equation(2) left side multiplied by A, we can get the equation below
    (3)A*A^T*A*u = A*e2*u = e2*A*u
    compare (1) with (3), if u is eigenvector of cov' of eigenvalue e, we can find that A*u = v
    (e is not zero, cov and cov' have the same not-zero eigenvalues, but have different number of zero eigenvalue, it can be proofed)
    so we can compute A^T*A instead of A*A^T to simplify the computation (which will generate a matrix with only 200 * 200 data)
    
    cov = np.matrix(diff.T)*np.matrix(diff) / total_face
    # compute the eigen values and eigen vectors of cov
    eigen_values, vectors = np.linalg.eigh(cov)

    eigen_vectors = np.matrix(diff)*np.matrix(vectors)
    # sort the eigenvalues and eigenvectors by desc
    sort_index = np.argsort(-eigen_values)
    eigen_values = eigen_values[sort_index]
    eigen_vectors = eigen_vectors[:, sort_index]
    print(eigen_values)
    '''

    # for each image we compute the y (y = A^T * x, weight) and we will compare yf(the input image) with yf, find the nearest one
    eigenfaces_weight = np.matrix(eigen_vectors.T)*np.matrix(diff)
    
    return img_mean, eigen_values, eigen_vectors, eigenfaces_weight