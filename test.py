import tensorflow as tf
import argparse
from base import config
import cv2
import numpy as np
import os


def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.compat.v1.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    print(return_elements[0])
    print(return_elements[1])
    print(return_elements[2])
    print(type(return_elements[0]))
    return return_elements


def test(path):

    graph = tf.Graph()
    return_tensors = read_pb_return_tensors(graph, config.PB_FILE, config.NODE_NAMES_1)

    for root, dirs, files in os.walk(path):
        for file in files:
            print(file)

            image = cv2.imread(os.path.join(root, file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(image.shape)

            image = np.array(image).reshape([-1, 28, 28, 1])

            with tf.Session(graph=graph) as sess:
                result = sess.run(return_tensors[2],
                                  feed_dict={return_tensors[0]: image})
                print(type(result))

            print(np.argmax(result, 1))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # parser.add_argument(
    #     '--image_path', type=str,
    #     default="./data/mnist/test/"
    # )
    # args = parser.parse_args()

    test("./data/mnist/test/")

