from cv2 import cv2
import numpy as np
import sys
import os
import PIL

train_sub_count = 40
train_img_count = 5
total_face = 200
row = 112
col = 92

def eigenfaces(src_path):
    img_list = np.empty((row*col, total_face))
    count = 0

    #read all the faces and flatten them
    for i in range(1, train_sub_count):
        for j in range(1, train_img_count):
            img_path = src_path + "/s" + str(i) + "/" + str(j) + ".pgm"
            #print(img_path)
            img = cv2.imread(img_path, 0)
            #cv2.imshow("1",img)
            img_col = np.array(img).flatten()

            img_list[:, count] = img_col[:]
            count += 1
    
    #compute the average of the faces
    img_mean = np.sum(img_list, axis=1) / total_face
    # average = img_mean.reshape(row, col).astype(np.uint8)
    # cv2.imshow("ave", average)
    # cv2.waitKey(0)

    #compute the delta
    for i in range(0, total_face):
        img_list[:, i] -= img_mean[:]
    

eigenfaces("D:\\k\\Myeigen\\MyEigenface\\att_faces")
    