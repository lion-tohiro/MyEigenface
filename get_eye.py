import sys
import os
from cv2 import cv2
import numpy as np

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if(f[-3:] == 'pgm'):
                fullname = os.path.join(root, f)
                yield fullname

def SetPoints(windowname, img):
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(temp_img, (x, y), 2, (255, 0, 0), -1)
            points.append([x, y])
            cv2.imshow(windowname, temp_img)

    temp_img = img.copy()
    cv2.imshow(windowname, temp_img)
    cv2.setMouseCallback(windowname, onMouse)
    key = cv2.waitKey(0)
    if key == 13:  # Enter
        print('point:', points)
        del temp_img
        cv2.destroyAllWindows()
        return points

base = 'D:\\k\\Myeigen\\MyEigenface\\att_faces\\s40'
# img = cv2.imread(base)
# point = SetPoints("src", img)
# point = np.array(point)
# eye_file = base.replace("pgm", "npy")
# np.save(eye_file, point)
for file in findAllFile(base):
    print(file)
    img = cv2.imread(file)
    point = SetPoints("src", img)
    point = np.array(point)
    eye_file = file.replace("pgm", "npy")
    np.save(eye_file, point)