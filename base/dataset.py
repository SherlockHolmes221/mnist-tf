from base import config
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class Dataset(object):
    def __init__(self, type):
        self.type = type
        self.mnist = input_data.read_data_sets("./data/mnist_data/", one_hot=True)
        if type == "train":
            self.data_path = config.TRAIN_DATA_PATH
            self.batch_size = config.TRAIN_BATCH_SIZE
            self.num_samples = config.TRAIN_NUM

        elif type == 'test':
            self.data_path = config.TEST_DATA_PATH
            self.batch_size = config.TEST_BATCH_SIZE
            self.num_samples = config.TEST_NUM

        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):

            if self.type == "train":
                if self.batch_count < self.num_batchs:
                    x, y = self.mnist.train.next_batch(self.batch_size)
                    self.batch_count += 1
                    return [x, y]
                else:
                    self.batch_count = 0
                    raise StopIteration

            elif self.type == 'test':
                if self.batch_count < self.num_batchs:
                    x, y = self.mnist.test.next_batch(self.batch_size)
                    self.batch_count += 1
                    return [x, y]
                else:
                    self.batch_count = 0
                    raise StopIteration

    def __len__(self):
        return self.num_batchs
