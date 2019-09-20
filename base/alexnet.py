from base import commom
import tensorflow as tf

class Alexnet(object):
    def __init__(self, input_x, input_y, class_num, trainable):
        print("Alex net")
        self.trainable = trainable
        try:
            self.predict = self.__build_network(input_x, class_num)
        except:
            raise NotImplementedError("Can not build up alex network!")
        self.y = input_y


    def __build_network(self, x_train, class_num):
        print("x_train.shape")
        print(x_train.shape)
        conv1 = commom.conv_relu(x_train, 1, 11, 11, 96, 1, 1, name='conv1', padding='VALID', trainable=self.trainable)
        lrn1 = commom.lrn(conv1, 2, 2e-05, 0.75, 'lrn1')
        pool1 = commom.max_pooling(lrn1, 2, 2, 1, 1, padding='VALID', name='pool1')

        print(pool1.shape)#17, 17,96

        conv2 = commom.conv_relu(pool1, int(pool1.get_shape()[-1]), 5, 5, 128, 1, 1, name='conv2', trainable=self.trainable)
        lrn2 = commom.lrn(conv2, 2, 2e-05, 0.75, 'lrn2')
        pool2 = commom.max_pooling(lrn2, 2, 2, 1, 1, name='pool2')

        print(pool2.shape)# 17, 17, 128

        # conv3 = commom.conv_relu(pool2, int(pool2.get_shape()[-1]), 3, 3, 384, 1, 1, name='conv3', trainable=self.trainable)
        #
        # conv4 = commom.conv_relu(conv3, int(conv3.get_shape()[-1]), 3, 3, 384, 1, 1, name='conv4', trainable=self.trainable)
        #
        # print(conv4.shape)#17, 17, 384
        #
        # conv5 = commom.conv_relu(conv4, int(conv4.get_shape()[-1]), 3, 3, 256, 1, 1, name='conv5', trainable=self.trainable)
        # pool5 = commom.max_pooling(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        #
        # print(pool5.shape)# 8, 8, 256

        flattened = tf.reshape(pool2, [-1, 17*17*128])
        fc6 = commom.fc(flattened, 17*17*128, 1024, name='fc6', trainable=self.trainable)
        dropout6 = commom.drop_out(fc6, 0.5)

        fc7 = commom.fc(dropout6, 1024, 512, name='fc7', trainable=self.trainable)
        dropout7 = commom.drop_out(fc7, 0.5)

        fc8 = commom.fc(dropout7, 512, class_num, relu=False, name='fc8', trainable=self.trainable)

        with tf.variable_scope('output'):
            predict = tf.nn.softmax(fc8, name="output")

        return predict

    def compute_loss(self):
        cross_entropy = -tf.reduce_sum(self.y*tf.log(self.predict))
        return cross_entropy

    def accuracy(self):
        correct_predict = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.y, 1))
        return tf.reduce_mean(tf.cast(correct_predict, tf.float32))
