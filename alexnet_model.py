import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

n_input = 784
n_classes = 10
dropout = 0.8

x = tf.placeholder(tf.float32, [None, 227 * 227 * 3])


def conv2d(name, l_input, w, b, k=1,pad='SAME'):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, k, k, 1], padding=pad),b), name=name)

def max_pool(name, l_input, k1, k2):
    return tf.nn.max_pool(l_input, ksize=[1, 3, 3, 1], strides=[1, k2, k2, 1], padding='VALID', name=name)

def norm(name, l_input, lsize=5):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):

    _X = tf.reshape(_X, shape=[-1, 227, 227, 3], name='input')


    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'], k=4, pad='VALID')

    act1  = tf.nn.relu(conv1, name='act1')

    pool1 = max_pool('pool1', act1, k1=2, k2=2)

    norm1 = norm('norm1', pool1, lsize=5)


    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])

    act2  = tf.nn.relu(conv2, name='act2')

    pool2 = max_pool('pool2', act2, k1=3, k2=2)

    norm2 = norm('norm2', pool2, lsize=5)


    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])

    act3  = tf.nn.relu(conv3, name='act3')

    conv4 = conv2d('conv4', act3, _weights['wc4'], _biases['bc4'])

    act4  = tf.nn.relu(conv4, name='act4')

    conv5 = conv2d('conv5', act4, _weights['wc5'], _biases['bc5'])

    act5  = tf.nn.relu(conv5, name='act5')

    pool5 = max_pool('pool5', act5, k1=3, k2=2)

    dense1 = tf.reshape(pool5, [-1, _weights['out'].get_shape().as_list()[0]])
    #dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')

    #dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation


    out = tf.matmul(dense1, _weights['out']) + _biases['out']
    output = tf.nn.softmax(out, name='output')
    print output
    return output


weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    #'wd1': tf.Variable(tf.random_normal([6 * 6 * 256, 4096])),
    #'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    #'out': tf.Variable(tf.random_normal([4096, 1000]))
    'out': tf.Variable(tf.random_normal([6 * 6 * 256, 1000]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    #'bd1': tf.Variable(tf.random_normal([4096])),
    #'bd2': tf.Variable(tf.random_normal([4096])),
    #'out': tf.Variable(tf.random_normal([1000]))
    'out': tf.Variable(tf.random_normal([1000]))
}



init = tf.initialize_all_variables()

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    pred = alex_net(x, weights, biases, 0.5)
    # Keep training until reach max iterations

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                    output_node_names=['output'])

    with tf.gfile.FastGFile('alex2gpu.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())