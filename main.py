#This code uses opencv to initialize the webcam and capture images
#images are converted to grayscale and saved to a local file
#edited by Lu Bilyk

import numpy as np
import cv2
from image_proc import define_Picture
from classify import nnetwork
import tensorflow as tf


capture = cv2.VideoCapture(0)

from pkg_resources import parse_version

OPCV3 = parse_version(cv2.__version__) >= parse_version('3')
interpret = "A"
accuracy = 92.4435

# Setup operations
with tf.device('/gpu:1'):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


def capPropId(prop):
  return getattr(cv2 if OPCV3 else cv2.cv,
    ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)

capture.set(capPropId("FRAME_WIDTH"), 640)
capture.set(capPropId("FRAME_HEIGHT"), 480)

while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()

    if ret == True:
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #standard = cv2.resize(gray, (320, 240))



        cv2.imwrite('gray_image.jpg', gray)

        # pass tbe image file to neural network to be tested against
        interpret, accuracy = nnetwork()

        cv2.putText(frame, str(interpret), (278, 430), cv2.FONT_HERSHEY_DUPLEX, 2.3, (0, 255, 0))
        cv2.putText(frame, "Accuracy: " + str(accuracy) + '%', (200, 460), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0))
        # Display the resulting frame
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
