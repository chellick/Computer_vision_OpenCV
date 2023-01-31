import cv2
import numpy as np
def onlyThisOne(image, lower, upper, kernel):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing


kernel = np.ones((5, 5), np.uint8)


LOWER = np.array([0, 0, 200])
UPPER = np.array([160, 160, 255])

image = cv2.imread("images/WIN_20221219_16_35_22_Pro.jpg")
image = cv2.resize(image, None, fx=0.5, fy=0.5)

new_image = onlyThisOne(image, LOWER, UPPER, kernel)

cv2.imshow("Yooohar", new_image)
cv2.waitKey(0)