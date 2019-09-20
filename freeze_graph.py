import tensorflow as tf
from base.alexnet import Alexnet
from base import config


with tf.name_scope('define_input'):
    input_x = tf.placeholder(dtype=tf.float32, name='input_x', shape=[None, 28, 28, 1])
    input_y = tf.placeholder(dtype=tf.float32, name='input_y', shape=[None, 10])
    trainable = tf.placeholder(dtype=tf.bool, name='training')


model = Alexnet(input_x, input_y, class_num=config.CLASS_NUM, trainable=trainable)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, config.CKPT_FILE)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def=sess.graph.as_graph_def(),
                            output_node_names=config.NODE_NAMES)

with tf.gfile.GFile(config.PB_FILE, "wb") as f:
    f.write(converted_graph_def.SerializeToString())




