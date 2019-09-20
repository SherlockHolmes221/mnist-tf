from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os
from base import config

mnist = input_data.read_data_sets("./data/mnist_data/", one_hot=True)
    
# for i in range(config.TRAIN_NUM):
#     image_array = mnist.train.images[i, :]
#     image_array = image_array.reshape(28, 28)
#     filename = save_dir_train + "mnist_train_%d.jpg" % i
#     array_to_image = scipy.misc.toimage(image_array, cmin = 0.0, cmax = 1.0,)
#     array_to_image.save(filename)
#

save_dir_test = config.TEST_DATA_PATH
for i in range(10):
    image_array = mnist.test.images[i, :]
    image_array = image_array.reshape(28, 28)
    filename = save_dir_test + "mnist_test_%d.jpg" % i
    array_to_image = scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0,)
    array_to_image.save(filename)
