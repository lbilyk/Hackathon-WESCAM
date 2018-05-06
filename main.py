#This code uses opencv to initialize the webcam and capture images
#images are converted to grayscale and saved to a local file
#edited by Lu Bilyk

import numpy as np
import cv2
from image_proc import define_Picture

capture = cv2.VideoCapture(0)

from pkg_resources import parse_version
OPCV3 = parse_version(cv2.__version__) >= parse_version('3')

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
        cv2.imwrite('gray_image.png', gray)
        cv2.putText(frame, "MASTER PROGRAMMER!!!!", (150, 440), cv2.FONT_HERSHEY_DUPLEX, .9, (50, 225, 50))
        # Display the resulting frame
        cv2.imshow('frame',gray)

        define_Picture(gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
