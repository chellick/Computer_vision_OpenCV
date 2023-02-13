
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np
import cv2
import keras



'''
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

class_names = ['zero', 'one', 'two', 'three', 'four', 'five',
               'six', 'seven', 'eight', 'nine']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

'''
#--------------------------------------------------------------------------------------------
'''
# tf.constant - константный тензор

a = tf.constant(1, shape=(1, 1))
b = tf.constant([1, 2, 3, 4])
c = tf.constant([[1, 2],
                [3, 4]], dtype=tf.float16)


a2 = tf.cast(c, dtype=tf.float32)

print(np.array(a2))
'''
#--------------------------------------------------------------------------------------------
'''
# tf.Variable - изменяемый массив
# tf.assign - заменить
# tf.assign_add - добавиь
# tf.assign_sub - убирать

# v1 = tf.Variable(-1.2)
# print(v1.assign_add(1))
#
#
# v2 = tf.Variable([3, 1, 2, 5], dtype=tf.float32)
# print(v2)
'''
#--------------------------------------------------------------------------------------------
'''
image = cv2.imread("C:\python\GitHub\CV\images\WIN_20221219_16_35_22_Pro.jpg")




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

new_image = onlyThisOne(image, LOWER, UPPER, kernel)

tensor_image = tf.constant(new_image)

plt.imshow(tensor_image, cmap=plt.cm.binary)
plt.show()
print(tensor_image)
#--------------------------------------------------------------------------------------------
# w = tf.Variable(tf.random.normal((3, 2)))
# b = tf.Variable(tf.zeros(2, dtype=tf.float32))
# x = tf.Variable([[-2.0, 1.0, 3.0]])
#
# with tf.GradientTape() as tape:
#     y = x @ w + b
#     # print(y, 'y')
#     loss = tf.reduce_mean(y ** 2)
#
#
# df = tape.gradient(loss, [w, b])
# print(df[0], df[1], sep='\n')

# Автоматическое дифференцирование функции
'''

from pixellib.instance import instance_segmentation

def obj_detect():
    s_image = instance_segmentation()
    s_image.load_model('\python\GitHub\CV\mask_rcnn_balloon.h5')    # TODO wrong file name correct

    target_class = s_image.select_target_classes(person=True)

    s_image.segmentImage(
        image_path="images\WIN_20221219_16_35_22_Pro.jpg",
        show_bboxes=True,
        segment_target_classes=target_class,
        output_image_name='output.jpg'
    )

def main():
    obj_detect()

if __name__ == '__main__':
    main()


