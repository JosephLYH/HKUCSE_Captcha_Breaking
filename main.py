import csv
from cv2 import cv2
import numpy as np
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import tensorflow as tf
import time

#---------------------------------------------------------------------------------------
# global variables
IMG_FILE = "image.png"
DATE = "30/04/2021"
TIME = "19:30"
END_TIME = "20:30"
TEAM = "Squash::1"
COURT = "FHS1"
COURTNO = { 
    "SHP2A": 1, "SHP2B": 2, "SHP2C": 3,
    "FHS1": 1, "FHS2": 2, "FHS3": 3, "FHS4": 4}

# account details
# UID = "calexlam"
# PASSWORD = "al26529833"

UID = "u3560501"
PASSWORD = "Cyanidebattery12"

#---------------------------------------------------------------------------------------
# functions
def preprocess(img):
    img_list = []
    
    # clip dark pixels to 0
    img = np.where(img < 240, 0, img)
    
    # straighten italic letters
    pts1 = np.float32([[11, 0],[135, 0],[0, 48],[124,48]])
    pts2 = np.float32([[0,0],[135, 0],[0,48],[135,48]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M ,(200, 50))
    
    # only letters are left after transformation
    # find boundary
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
            
    # crop boundary
    img = img[:, left_bound:right_bound+1]
    
    cols = [0]*7
    cols[6] = img.shape[1]
    
    # take right 3 numbers with 20 px
    for i in range(3):
        cols[5-i] = img.shape[1] - (i+1)*20

    # split remaining letters into 3 segments
    for i in range(2):
        cols[i+1] = (i+1)*((img.shape[1]-60)//3)

    for i in range(6):
        img_cropped = img[:, cols[i]:cols[i+1]]
        
        # resize to 50x30 px
        width = 30
        height = img_cropped.shape[0] # keep original height
        dim = (width, height)
        img_cropped = cv2.resize(img_cropped, dim, interpolation = cv2.INTER_AREA)
        
        # normalize to 0-1
        img_cropped = img_cropped/255

        # convert to float32
        img_cropped = img_cropped.astype(np.float32)
        
        img_list.append(img_cropped)
    
    img_list = np.asarray(img_list)
    img_list = np.expand_dims(img_list, axis=3)
    
    return img_list
        
def prediction(img):
    NUMBERS = '345678'
    LETTERS = 'bcdefghkmnprwxy'
    labels = np.array(list(NUMBERS + LETTERS))
    
    img_list = preprocess(img)
    pred_labels = labels[np.argmax(model.predict(img_list), axis=1)]
    prediction = ''.join(pred_labels)
    
    return prediction

#---------------------------------------------------------------------------------------
# main

# use GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# use CPU instead
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load model
model = tf.keras.models.load_model('captcha/captcha_cnn')

# start driver
driver = webdriver.Chrome()

driver.get("https://hkuportal.hku.hk/login.html")
login_form = driver.find_element_by_name("form")
uid_input = driver.find_element_by_id("username")
password_input = driver.find_element_by_id("password")

# login to portal
uid_input.send_keys(UID)
password_input.send_keys(PASSWORD)
login_form.submit()

# switch to page
driver.get("https://sis-eportal.hku.hk/psp/ptlprod_newwin/EMPLOYEE/EMPL/e/?url=https%3a%2f%2fbs.cse.hku.hk&FolderPath=PORTAL_ROOT_OBJECT.Z_CAMPUS_INFORMATION_SERVICES.Z_N_SERVICE_DEPARTMENTS.Z_N_IHP.Z_N_SPORTS_FACILITIES_BOOKING&IsFolder=false&IgnoreParamTempl=FolderPath%2cIsFolder")
driver.get("https://bs.cse.hku.hk/ihpbooking/servlet/IHP_Booking/showActivityList")
driver.execute_script("javascript:goToActivity('{}')".format(TEAM))
driver.execute_script("javascript:startBooking('{}')".format(DATE))
driver.execute_script("javascript:bookThisSlot('{}', '{}', '{}')".format(TIME, COURT, COURTNO[COURT]))
driver.switch_to.alert.accept()

# select booking (first booking option) and end time
try:
    driver.find_element_by_xpath("/html/body/form/table/tbody/tr[4]/td[2]/input[1]").click()
    Select(driver.find_element_by_name("end_time")).select_by_visible_text(END_TIME)
    driver.find_element_by_name("form").submit()

    # agree page
    if TEAM == "Squash::1":
        driver.find_element_by_name("no_of_people").send_keys("2")
    driver.find_element_by_name("agree").click()
    driver.find_element_by_name("form").submit()

    # captcha page (only uncomment when book)
    # while True:
    #     try:
    #         driver.find_element_by_name("imgCaptcha").screenshot(IMG_FILE)
    #         driver.find_element_by_id("ansCaptcha").send_keys(prediction(cv2.imread(IMG_FILE, 0)))
    #         driver.find_element_by_id("btnSubmit").click()
    #     except:
    #         quit()

except:
    print('Time already booked!')
    quit()