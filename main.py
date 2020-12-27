import os
import sys
from cv2 import cv2
import numpy as np
from eigenface import eigenfaces_train
from eigenface_detect import eigenfaces_detect
from eigenface_restruct import eigenfaces_restruct
from base import normalize

train = True

# some parameters of training and testing data
train_sub_count = 40
train_img_count = 5
total_face = 200
row = 70
col = 70

model_path = "D:\\k\\Myeigen\\MyEigenface\\model.npz"

if train == True:
    average, values, vectors, weight = eigenfaces_train("D:\\k\\Myeigen\\MyEigenface\\att_faces")

    np.savez(model_path, average=average, values=values, vectors=vectors, weight=weight)

    # show the average face
    average_img = average.reshape(row, col).astype(np.uint8)
    cv2.imshow("average-faces", average_img)

    # show the top 10 eigenfaces
    eigenfaces_img = np.empty((row, col*10), dtype=np.uint8)
    for i in range(10):
        eigenfaces_img_temp = vectors.T[i].reshape(row, col)
        eigenfaces_img_temp = normalize(eigenfaces_img_temp)
        eigenfaces_img[:, col*i:col*(i+1)] = eigenfaces_img_temp
    cv2.imshow("eigenfaces",eigenfaces_img)

    cv2.waitKey(0)
else:
    data = np.load(model_path)
    average = data["average"]
    values = data["values"]
    vectors = data["vectors"]
    weight = data["weight"]

    # show the average face
    average_img = average.reshape(row, col).astype(np.uint8)
    cv2.imshow("average-faces", average_img)

    # show the top 10 eigenfaces
    eigenfaces_img = np.empty((row, col*10), dtype=np.uint8)
    for i in range(10):
        eigenfaces_img_temp = vectors.T[i].reshape(row, col)
        eigenfaces_img_temp = normalize(eigenfaces_img_temp)
        eigenfaces_img[:, col*i:col*(i+1)] = eigenfaces_img_temp
    cv2.imshow("eigenfaces",eigenfaces_img)

    eigenfaces_detect("D:\\k\\Myeigen\\MyEigenface\\att_faces\\s6\\6", average, values, vectors, weight, 0.95)

    eigenfaces_restruct("D:\\k\\Myeigen\\MyEigenface\\att_faces\\s34\\8", average, values, vectors)

    cv2.waitKey(0)