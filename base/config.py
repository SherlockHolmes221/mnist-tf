MNIST_DATA_PATH = './data/mnist'
TRAIN_NUM = 50000
TEST_NUM = 10000
CLASS_NUM = 10

TRAIN_DATA_PATH = './data/mnist/train'
TRAIN_BATCH_SIZE = 50
TRAIN_LI_FIRST = 0.01
TRAIN_LI_END = 0.001


TEST_DATA_PATH = './data/mnist/test'
TEST_BATCH_SIZE = 20

TRAIN_EPOCH = 1


PB_FILE = "./model/mnist_tf.pb"
CKPT_FILE = "./output/checkpoint/minist_test_loss=1.7715.ckpt-0"

NODE_NAMES = ["input/input_x", "input/input_y", "output/output"]
NODE_NAMES_1 = ["input/input_x:0", "input/input_y:0", "output/output:0"]
