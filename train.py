import tensorflow as tf
from base import config
from tqdm import tqdm
from base.dataset import Dataset
from base.alexnet import Alexnet
import numpy as np
import time
import os
import shutil


class MnistTrain(object):

    def __init__(self):
        print('MnistTrain_init_begin')
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.epoch = config.TRAIN_EPOCH
        self.train_dataset = Dataset('train')
        self.test_dataset = Dataset('test')

        with tf.name_scope('define_input'):
            self.input_x = tf.placeholder(dtype=tf.float32, name='input_x', shape=[None, 28, 28, 1])
            self.input_y = tf.placeholder(dtype=tf.float32, name='input_y', shape=[None, 10])
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss_acc"):
            self.model = Alexnet(self.input_x, self.input_y, config.CLASS_NUM, self.trainable)
            self.net_var = tf.global_variables()
            self.loss = self.model.compute_loss()
            self.accuracy = self.model.accuracy()

        with tf.name_scope('learn_rate'):
            self.learn_rate = config.TRAIN_LI_FIRST

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        with tf.name_scope('saver'):
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("total_loss", self.loss)

            logdir = "./output/log/"
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(logdir, graph=self.sess.graph)

        print('MnistTrain_init_end')

    def train(self):
        print('MnistTrain_train_begin')

        self.sess.run(tf.global_variables_initializer())

        for step in range(self.epoch):
            pbar = tqdm(self.train_dataset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                print(len(pbar))
                _, train_step_loss, train_step_accuracy = self.sess.run([self.optimizer, self.loss, self.accuracy], feed_dict={
                    self.input_x: np.array(train_data[0]).reshape([config.TRAIN_BATCH_SIZE, 28, 28, 1]),
                    self.input_y: np.array(train_data[1]).reshape([config.TRAIN_BATCH_SIZE, 10]),
                    self.trainable: True,
                })
                train_epoch_loss.append(train_step_loss)
                pbar.set_description("train loss: %.2f" % train_step_loss + " train accuracy: %.2f" % train_step_accuracy)

            print("train_end")

            for test_data in self.test_dataset:
                _, test_step_loss, test_step_accuracy = self.sess.run([self.optimizer, self.loss, self.accuracy], feed_dict={
                    self.input_x: np.array(test_data[0]).reshape([config.TEST_BATCH_SIZE, 28, 28, 1]),
                    self.input_y: np.array(test_data[1]).reshape([config.TEST_BATCH_SIZE, 10]),
                    self.trainable: False,
                })
                print("test loss: %.2f" % test_step_loss + " test accuracy: %.2f" % test_step_accuracy)
                test_epoch_loss.append(test_step_loss)

            print("test_end")

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./output/checkpoint/minist_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(step, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=step)

            print('MnistTrain_train_end')


if __name__ == '__main__':
    MnistTrain().train()


