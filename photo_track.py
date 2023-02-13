'''
import cv2
import numpy as np
def onlyThisOne(image, lower, upper, kernel):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing


kernel = np.ones((6, 6), np.uint8)


LOWER = np.array([0, 0, 200])
UPPER = np.array([160, 160, 255])

image = cv2.imread("images/WIN_20221219_16_38_00_Pro.jpg")
image = cv2.resize(image, None, fx=0.5, fy=0.5)

new_image = onlyThisOne(image, LOWER, UPPER, kernel)

# print(type(new_image))

cv2.imshow("Переход", new_image)
cv2.waitKey(0)
'''


import cv2
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

path = 'C:\\python\\GitHub\\CV\\images'
# print(os.listdir(path))

data = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0])
img_data = []
temp_data = []
train_tensor = []


for g, i in enumerate(os.listdir(path)):
    image = cv2.imread('images' + '/' + i)

    def onlyThisOne(image, lower, upper, kernel):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower, upper)

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        return closing


    kernel = np.ones((6, 6), np.uint8)

    LOWER = np.array([0, 0, 200])
    UPPER = np.array([160, 160, 255])

    new_image = onlyThisOne(image, LOWER, UPPER, kernel)

    # tensor_image = tf.constant(new_image)


    train_tensor.append(new_image)
    # temp_data.append(i)
    # temp_data.append(tensor_image)
    # temp_data.append(data[g])
    #
    # img_data.append(temp_data)
    # temp_data = []







path = 'C:\\python\\GitHub\\CV\\cut'

img_data_c = []
temp_data_c = []
test_tensor = []
data_c = np.array([0, 1, 0, 1, 0])

for g, i in enumerate(os.listdir(path)):
    image = cv2.imread('cut' + '/' + i)


    def onlyThisOne(image, lower, upper, kernel):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower, upper)

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        return closing


    kernel = np.ones((6, 6), np.uint8)

    LOWER = np.array([0, 0, 200])
    UPPER = np.array([160, 160, 255])

    new_image = onlyThisOne(image, LOWER, UPPER, kernel)

    # tensor_image = tf.constant(new_image)

    test_tensor.append(new_image)

    # temp_data_c.append(i)
    # temp_data_c.append(tensor_image)
    #
    # img_data_c.append(temp_data_c)
    # temp_data_c = []


print(len(train_tensor))
print(len(data))



model = keras.Sequential([keras.layers.Flatten(input_shape=(1920, 1080)),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(2)])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_tensor, data, epochs=5)
test_loss, test_acc = model.evaluate(test_tensor, data_c)





