import tensorflow as tf
import argparse
from base import config
import cv2
import numpy as np


def test(path):
    print(path)
    image = cv2.imread(path)
    print(image.shape)
    image = np.array(image).reshape([-1, 28, 28, 1])

    with tf.Session(graph= tf.Graph()) as sess:
        result = sess.run(
        config.NODE_NAMES[1], feed_dict={config.NODE_NAMES[0]: image})

    print(result)
    return tf.argmax(result, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--image_path', type=str,
    )
    args = parser.parse_args()

    if 'image_path' in args:
        result = test(args.image_path)
        print(result)
