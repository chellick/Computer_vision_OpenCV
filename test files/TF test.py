
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np

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

# tf.Variable - изменяемый массив
# tf.assign - заменить
# tf.assign_add - добавиь
# tf.assign_sub - убирать

v1 = tf.Variable(-1.2)
print(v1.assign_add(1))


v2 = tf.Variable([3, 1, 2, 5], dtype=tf.float32)
print(v2)

