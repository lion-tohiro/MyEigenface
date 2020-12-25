from cv2 import cv2
import numpy as np
import sys
import os

# some parameters of training and testing data
train_sub_count = 40
train_img_count = 5
total_face = 200
row = 112
col = 92

def eigenfaces_train(src_path, a):
    img_list = np.empty((row*col, total_face))
    count = 0

    # read all the faces and flatten them
    for i in range(1, train_sub_count):
        for j in range(1, train_img_count):
            img_path = src_path + "/s" + str(i) + "/" + str(j) + ".pgm"
            # print(img_path)
            img = cv2.imread(img_path, 0)

            img_col = np.array(img).flatten()
            img_list[:, count] = img_col[:]
            count += 1
    
    # compute the average of the faces
    img_mean = np.sum(img_list, axis=1) / total_face

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
    # normalized the matrix
    eigenfaces_mat = 255*(eigenfaces_mat - np.min(eigenfaces_mat)) / (np.max(eigenfaces_mat) - np.min(eigenfaces_mat))

    # compute the weight of trainging data
    eigenfaces_weight = np.matrix(img_list.T)*np.matrix(eigenfaces_mat)
    # print(np.size(eigenfaces_weight))
    
    return img_mean, eigenfaces_mat, eigenfaces_weight

def eigenfaces_detect(src, average, eigenfaces, weight):
    src_img = cv2.imread(src, 0)
    img_col = np.array(src_img).flatten()

    diff = img_col-average
    src_weight = np.matrix(diff.T)*np.matrix(eigenfaces)

    dis = []

    for i in range(total_face):
        distance = np.linalg.norm(weight[i]- src_weight)
        dis.append(distance)
    
    # get the index of detected face
    index = np.argsort(dis)
    # print(index)

    # get the path of the detected face
    # get the subject dir
    sub = (int)((index[0] + 1) / 5 + 1)
    # print(sub)
    # get the number of the face in a certain subject
    number = (index[0] + 1) % 5
    # print(number)
    # merge the subject and number to the path
    dist = "D:\\k\\Myeigen\\MyEigenface\\att_faces\\s" + str(sub) + "/" + str(number) + ".pgm"
    dist_img = cv2.imread(dist, 0)

    detect = np.empty((row, col*2), dtype=np.uint8)
    detect[:, 0:col] = src_img
    detect[:, col:col*2] = dist_img
    cv2.imshow("detect", detect)

# the second parameter is always between 0.95-0.99
average, eigenfaces, weight = eigenfaces_train("D:\\k\\Myeigen\\MyEigenface\\att_faces", 0.95)

# show the average face
average_img = average.reshape(row, col).astype(np.uint8)
cv2.imshow("average-faces", average_img)

# show the top 10 eigenfaces
eigenfaces_img = np.empty((row, col*10), dtype=np.uint8)
for i in range(10):
    eigenfaces_img_temp = eigenfaces.T[i].reshape(row, col)
     # cv2.imshow(str(i), eigenface_img_temp)
    eigenfaces_img[:, col*i:col*(i+1)] = eigenfaces_img_temp
cv2.imshow("eigenfaces",eigenfaces_img)

# do detection
eigenfaces_detect("D:\\k\\Myeigen\\MyEigenface\\att_faces\\s1\\6.pgm", average, eigenfaces, weight)

cv2.waitKey(0)