import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import cv2 as cv



def define_Picture(img):
    string = 0
    maxValue = 34
    image3 = 0

    mask = cv.imread("C:\\Users\L_bil\\Documents\\Hackathon", 0)
    lower = np.array([0, 30, 40], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    image3 = cv.resize(img, (512, 512))

    converted = (cv.cvtColor(image3, cv.COLOR_BGR2HSV))
    skinmask = cv.inRange(converted, lower, upper)
    res = cv.bitwise_and(image3, image3, mask = skinmask)
    plt.imshow(res)
    plt.show()

    ret, thresh = cv.threshold(res, 250, 255, cv.THRESH_BINARY, 0)

    plt.imshow(thresh)

    res = cv.Canny(res, 300, 100, 0)
    test = np.array(res)

    print(test)
    plt.imshow(res)
    plt.show()


