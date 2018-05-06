import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import cv2 as cv



def define_Picture(img):
    string = 0
    maxValue = 34
    image3 = 0

    mask = cv.imread("C:/Users\L_bil/Documents/Hackathon/gray block.png", 0)
    lower = np.array([0, 30, 40], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    image3 = cv.resize(img, (299, 299))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    converted = (cv.cvtColor(image3, cv.COLOR_BGR2HSV))
    skinmask = cv.inRange(converted, lower, upper)
    res = cv.bitwise_and(image3, image3, mask=skinmask)
    plt.imshow(res)
    plt.show()



    ret, thresh = cv.threshold(res, 125, 0, cv.THRESH_BINARY, 0)

    plt.imshow(gray)
    plt.show()

    res = cv.Canny(res, 100, 100, 0)
    test = np.array(gray)

    print(test)
    plt.imshow(gray)
    plt.show()


