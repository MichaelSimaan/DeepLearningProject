import tensorflow as tf

class Model:
    def __init__(self):
        self.var_list = []
        self.allvars = []
        self.optimizer = None
        self.senoptimizer = None


    def getModel(self):
        _IMAGE_SIZE = 32
        _IMAGE_CHANNELS = 3
        _NUM_CLASSES = 10
        dropout = 0.25
        n_classes = 10

        conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
        conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
        conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
        conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

        with tf.name_scope('main_params'):
            x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
            y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
            x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

            global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
            learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        with tf.variable_scope('conv1') as scope:
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(x_image, 64, (3, 3), use_bias=False, padding='SAME', name="layer1", trainable=True)
            conv1 = tf.layers.batch_normalization(conv1, training=True,name="layer2")
            conv1 = tf.nn.relu(conv1,name="layer3")
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.average_pooling2d(conv1, 2, 2, name="layer4")

        with tf.variable_scope('conv2') as scope:
            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 128, (3, 3), use_bias=False, padding='SAME',name="layer4")
            conv2 = tf.layers.batch_normalization(conv2, training=True,name="layer5")
            conv2 = tf.nn.relu(conv2)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.average_pooling2d(conv2, 2, 2)

        with tf.variable_scope('conv3') as scope:
            # Convolution Layer with 64 filters and a kernel size of 3
            conv3 = tf.layers.conv2d(conv2, 256, (3, 3), use_bias=False, padding='SAME')
            conv3 = tf.layers.batch_normalization(conv3, training=True)
            conv3 = tf.nn.relu(conv3)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv3 = tf.layers.average_pooling2d(conv3, 2, 2)

        with tf.variable_scope('conv4') as scope:
            # Convolution Layer with 64 filters and a kernel size of 3
            conv4 = tf.layers.conv2d(conv3, 728, (3, 3), use_bias=False, padding='SAME', name='layer4')
            conv4 = tf.layers.batch_normalization(conv4, training=True)
            conv4 = tf.nn.relu(conv4)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv4 = tf.layers.average_pooling2d(conv4, 2, 2)

        with tf.variable_scope('Flaten_and_dense') as scope:
            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv4)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, 1024)
            # Apply Dropout (if is_training is False, dropout is not applied)
            fc1 = tf.layers.dropout(fc1, rate=dropout)

            # Output layer, class prediction
            softmax = tf.layers.dense(fc1, n_classes,name="last")
        y_pred_cls = tf.argmax(softmax, axis=1)
        return x, y, softmax, y_pred_cls, global_step, learning_rate

    def lr(self, epoch):
        learning_rate = 1e-3
        if epoch > 80:
            learning_rate *= 0.5e-3
        elif epoch > 60:
            learning_rate *= 1e-3
        elif epoch > 40:
            learning_rate *= 1e-2
        elif epoch > 20:
            learning_rate *= 1e-1
        return learning_rate

    def freeze(self, variable, name):
        tempList = []
        varname = variable + "/" + name
        for var in self.var_list:
            if varname not in var.name:
                tempList.append(var)
        self.var_list = tempList

    def freezeAllExcept(self, variable, name):
        tempList = []
        varname = variable + "/" + name
        for var in self.var_list:
            if varname in var.name:
                tempList.append(var)
        self.var_list = tempList

    def reset(self, variable, name, session):
        varname = variable + "/" + name
        for var in self.allvars:
            if varname in var.name:
                session.run(var.initializer)
