import csv
from cv2 import cv2
import numpy as np
import os
import pandas as pd
import shutil

SOURCEPATH = 'data_bw'
SAVEPATH = 'data_letter'

CHARACTERS = '345678bcdefghkmnprwxy'

char_count = {}
for char in CHARACTERS:
    char_count[char] = 0

if os.path.exists(SAVEPATH):
    shutil.rmtree(SAVEPATH)
os.mkdir(SAVEPATH)
for char in CHARACTERS:
    if not os.path.exists(os.path.join(SAVEPATH, char)):
        os.mkdir(os.path.join(SAVEPATH, char))

csv_arr = pd.read_csv('data_labels.csv', delimiter=',', names=['fname','chars'])

for file in os.listdir(SOURCEPATH):
    fname = int(os.path.splitext(file)[0])

    if fname in csv_arr['fname']:
        # locate image
        img = cv2.imread(os.path.join(SOURCEPATH, file), 0)

        cols = [0]*7
        cols[6] = img.shape[1]

        # take right 3 numbers with 20 px
        for i in range(3):
            cols[5-i] = img.shape[1] - (i+1)*20
        
        # split remaining letters into 3 segments
        for i in range(2):
            cols[i+1] = (i+1)*((img.shape[1]-60)//3)

        for i, char in enumerate(csv_arr[csv_arr['fname'] == fname]['chars'].values[0]):
            char_count[char] += 1
            
            # uncomment for debugging mislabelled data
            # if char == 'h' and char_count[char] == 25:
            #     print(fname, csv_arr[csv_arr['fname'] == fname]['chars'].values)
            
            # create processed image
            cv2.imwrite(os.path.join(SAVEPATH, char, str(char_count[char]) + '.jpg'), img[:, cols[i]:cols[i+1]])