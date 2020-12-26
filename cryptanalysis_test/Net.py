import tensorflow as tf

def _conv1D(input, filter, stride, kernelSize, name, activation = tf.nn.sigmoid):
    with tf.variable_scope(name):
        return tf.layers.conv1d(inputs=input, filters=filter, strides=stride,
                                   kernel_size=kernelSize, padding='SAME', activation=activation, use_bias=False)

def _ConvNet(input, unitsLength):

    input = tf.convert_to_tensor(input, dtype=tf.float32)
    input = tf.reshape(input, shape=[-1, unitsLength, 1])
    # print(input.shape)
    with tf.name_scope('convlayers'):
        conv1 = _conv1D(input, 2, 1, [4], name='conv_1')
        conv2 = _conv1D(conv1, 4, 2, [2], name='conv_2')
        conv3 = _conv1D(conv2, 4, 1, [1], name='conv_3')
        output = _conv1D(conv3, 1, 1, [1], name='conv_4', activation=tf.nn.tanh)
        return output

def _build_Network(plain, key, predict_key, plainTextLength, keyLength):
    unitsLength = plainTextLength + keyLength

    with tf.variable_scope('Alice'):
        Alice_input = tf.concat([plain, key], axis=1)
        A_w = tf.Variable(tf.truncated_normal(shape=[unitsLength, unitsLength], mean=0, stddev=0.1))
        Alice_FC_layer = tf.nn.sigmoid(tf.matmul(Alice_input, A_w))
        Alice_output = _ConvNet(Alice_FC_layer, unitsLength)
 
    reshape_Alice_output = tf.reshape(Alice_output, shape=[-1, plainTextLength])
    
    with tf.variable_scope('Bob'):
        Bob_input = tf.concat([reshape_Alice_output, key], axis=1)
        B_w = tf.Variable(tf.truncated_normal(shape=[unitsLength, unitsLength], mean=0, stddev=0.1))
        Bob_FC_layer = tf.nn.sigmoid(tf.matmul(Bob_input, B_w))
        Bob_output = _ConvNet(Bob_FC_layer, unitsLength)

    with tf.variable_scope('Eve'):
        E_w_1 = tf.Variable(tf.truncated_normal(shape=[plainTextLength, unitsLength], mean=0, stddev=0.1))
        E_FC_layer1 = tf.nn.sigmoid(tf.matmul(reshape_Alice_output, E_w_1))
        E_w_2 = tf.Variable(tf.truncated_normal(shape=[unitsLength, unitsLength], mean=0, stddev=0.1))
        E_FC_layer2 = tf.nn.sigmoid(tf.matmul(E_FC_layer1, E_w_2))
        Eve_output = _ConvNet(E_FC_layer2, unitsLength)
    
    with tf.variable_scope('Eve1'):
        Eve1_input = tf.concat([reshape_Alice_output, predict_key[:,0:8]], axis=1)
        E1_w_1 = tf.Variable(tf.truncated_normal(shape=[plainTextLength + 8, unitsLength], mean=0, stddev=0.1))
        E1_FC_layer1 = tf.nn.sigmoid(tf.matmul(Eve1_input, E1_w_1))
        E1_w_2 = tf.Variable(tf.truncated_normal(shape=[unitsLength, unitsLength], mean=0, stddev=0.1))
        E1_FC_layer2 = tf.nn.sigmoid(tf.matmul(E1_FC_layer1, E1_w_2))
        Eve1_output = _ConvNet(E1_FC_layer2, unitsLength)

    return Alice_output, Bob_output, Eve_output, Eve1_output