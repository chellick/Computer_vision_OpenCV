'''
from cvzone.PoseModule import PoseDetector
import cv2
from matplotlib import pyplot as pltd
# cap = cv2.VideoCapture(0)
# detector = PoseDetector()
#
# while True:
#     success, img = cap.read()
#     img = detector.findPose(img)
#     lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)
#     if bboxInfo:
#         center = bboxInfo["center"]
#         cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
#
#     cv2.imshow("Image", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


import cv2
import imutils

# Initializing the HOG person
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


image = cv2.imread('images/WIN_20221219_16_38_00_Pro.jpg')

image = imutils.resize(image,
                       width=min(2000, image.shape[1]))

(regions, _) = hog.detectMultiScale(image,
                                    winStride=(10, 20),
                                    padding=(0, 0),
                                    scale=1.05)

for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y),
                  (x + w, y + h),
                  (255, 255, 255), 2)


cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()

'''

import cv2
from matplotlib import pyplot as plt

# Opening image
img = cv2.imread("images/WIN_20221219_16_38_00_Pro.jpg")

# OpenCV opens images as BRG
# but we want it as RGB We'll
# also need a grayscale version
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Use minSize because for not
# bothering with extra-small
# dots that would look like STOP signs
stop_data = cv2.CascadeClassifier('stop_data.xml')

found = stop_data.detectMultiScale(img_gray,
                                   minSize=(20, 20))

# Don't do anything if there's
# no sign
amount_found = len(found)

if amount_found != 0:

    # There may be more than one
    # sign in the image
    for (x, y, width, height) in found:
        # We draw a green rectangle around
        # every recognized sign
        cv2.rectangle(img_rgb, (x, y),
                      (x + height, y + width),
                      (0, 255, 0), 5)

# Creates the environment of
# the picture and shows it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()