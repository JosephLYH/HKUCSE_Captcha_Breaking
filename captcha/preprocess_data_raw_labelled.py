from cv2 import cv2
import numpy as np
import os
import shutil

SOURCEPATH = 'data_raw_labelled'
SAVEPATH = 'data_bw'

if os.path.exists(SAVEPATH):
    shutil.rmtree(SAVEPATH)
os.mkdir(SAVEPATH)

for file in os.listdir(SOURCEPATH):
    # locate image
    img = cv2.imread(os.path.join(SOURCEPATH, file), 0)

    # clip dark pixels to 0
    img = np.where(img < 240, 0, img)
    
    # straighten italic letters
    pts1 = np.float32([[11, 0],[135, 0],[0, 48],[124,48]])
    pts2 = np.float32([[0,0],[135, 0],[0,48],[135,48]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M ,(200, 50))

    # only letters are left after transformation
    # crop boundary
    left_bound = 0
    right_bound = img.shape[1]

    for i in range(200):
        if img[:,i].max() > 150:
            left_bound = i
            break
    
    for i in range(200)[::-1]:
        if img[:,i].max() > 150:
            right_bound = i
            break

    # exaggerate characters features
    # img = np.where(img > 0, 255, img)

    # create processed image
    cv2.imwrite(os.path.join(SAVEPATH, file), img[:, left_bound:right_bound+1])