import os
import utils
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import imageio

# --- set parameters
LEARNING_RATE = 0.0002
BETA1 = 0.5
BATCH_SIZE = 64
EPOCHS = 2
SAVE_SAMPLE_PERIOD = 50

# --- folder to save samples
if not os.path.exists('samples'):
    os.mkdir('samples')


def lrelu(x, alpha=0.2):
    """ Leaky ReLU activation """
    return tf.maximum(alpha * x, x)


class ConvLayer:
    def __init__(self, name, mi, mo, apply_batch_norm, filter_size=5, stride=2, f=tf.nn.relu):
        """
        Convolution layer for the DCGAN
        :param name: parameter (weight) name
        :param mi: input feature map size
        :param mo: output feature map size
        :param apply_batch_norm: boolean flag whether to apply batch normalization
        :param filter_size: convolution filter size
        :param stride: stride size
        :param f: activation function
        """

        # --- create weight variable
        self.W = tf.get_variable(
            "W_%s" % name,
            shape=(filter_size, filter_size, mi, mo),
            initializer=tf.truncated_normal_initializer(stddev=0.02),
        )

        # --- create bias variable
        self.b = tf.get_variable(
            "b_%s" % name,
            shape=(mo,),
            initializer=tf.zeros_initializer(),
        )

        self.name = name
        self.f = f
        self.stride = stride
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        # --- apply the convolution
        conv_out = tf.nn.conv2d(
            X,
            self.W,
            strides=[1, self.stride, self.stride, 1],
            padding='SAME'
        )

        # --- add the bias
        conv_out = tf.nn.bias_add(conv_out, self.b)

        # --- do batch normalization
        with tf.variable_scope(self.name) as scope:
            if self.apply_batch_norm:
                conv_out = tf.layers.batch_normalization(
                    conv_out,
                    momentum=0.9,
                    epsilon=1e-5,
                    scale=True,
                    training=is_training,
                    reuse=reuse,
                )
        return self.f(conv_out)


class FractionallyStridedConvLayer:
    def __init__(self, name, mi, mo, output_shape, apply_batch_norm, filter_size=5, stride=2, f=tf.nn.relu):
        """
        The same as ConvLayer, but the input and output feature map sizes are switched
        in the convolutional filter
        :param name: parameter (weight) name
        :param mi: input feature map size
        :param mo: output feature map size
        :param output_shape: output shape
        :param apply_batch_norm: boolean flag whether to apply batch normalization
        :param filter_size: convolution filter size
        :param stride: stride size
        :param f: activation function
        """

        # --- create weight variable
        self.W = tf.get_variable(
            "W_%s" % name,
            shape=(filter_size, filter_size, mo, mi),
            initializer=tf.random_normal_initializer(stddev=0.02),
        )

        # --- create bias variable
        self.b = tf.get_variable(
            "b_%s" % name,
            shape=(mo,),
            initializer=tf.zeros_initializer(),
        )

        self.f = f
        self.stride = stride
        self.name = name
        self.output_shape = output_shape
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        # --- apply the convolution ("transposed")
        conv_out = tf.nn.conv2d_transpose(
            value=X,
            filter=self.W,
            output_shape=self.output_shape,
            strides=[1, self.stride, self.stride, 1],
        )
        conv_out = tf.nn.bias_add(conv_out, self.b)

        # --- do batch normalization
        with tf.variable_scope(self.name) as scope:
            if self.apply_batch_norm:
                conv_out = tf.layers.batch_normalization(
                    conv_out,
                    momentum=0.9,
                    epsilon=1e-5,
                    scale=True,
                    training=is_training,
                    reuse=reuse,
                )

        return self.f(conv_out)


class DenseLayer(object):
    """ Dense layer """
    def __init__(self, name, M1, M2, apply_batch_norm, f=tf.nn.relu):
        self.W = tf.get_variable(
            "W_%s" % name,
            shape=(M1, M2),
            initializer=tf.random_normal_initializer(stddev=0.02),
        )
        self.b = tf.get_variable(
            "b_%s" % name,
            shape=(M2,),
            initializer=tf.zeros_initializer(),
        )
        self.f = f
        self.name = name
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        a = tf.matmul(X, self.W) + self.b

        # --- apply batch normalization
        with tf.variable_scope(self.name) as scope:
            if self.apply_batch_norm:
                a = tf.layers.batch_normalization(
                    a,
                    momentum=0.9,
                    epsilon=1e-5,
                    scale=True,
                    training=is_training,
                    reuse=reuse,
                )
        return self.f(a)


class DCGAN:
    def __init__(self, img_dim, num_colors, d_sizes, g_sizes):

        self.img_dim = img_dim
        self.num_colors = num_colors
        self.latent_dims = g_sizes['z']
        self.g_sizes = g_sizes
        self.d_sizes = d_sizes

        # --- layers-related parameters defined later
        self.d_convlayers = None
        self.d_denselayers = None
        self.d_finallayer = None
        self.g_dims = None
        self.g_denselayers = None
        self.g_convlayers = None

        # --- input data: real images
        self.X = tf.placeholder(
            tf.float32,
            shape=(None, img_dim, img_dim, num_colors),
            name='X'
        )

        # --- input data: random vectors
        self.Z = tf.placeholder(
            tf.float32,
            shape=(None, self.latent_dims),
            name='Z'
        )

        # --- by making batch_size a placeholder, can specify a variable
        #     number of samples in the FS-conv operation
        self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_sz')

        # --- build the discriminator (returns predictions for the inputs)
        logits = self.build_discriminator()

        # --- build the generator (generates sample images)
        self.sample_images = self.build_generator()

        # --- pass the sample images through the discriminator to get sample logits
        #     to get the discriminator cost
        with tf.variable_scope("discriminator") as scope:
            scope.reuse_variables()  # so that the batch norm will reuse the already created variables
            sample_logits = self.d_forward(self.sample_images, True)

        # --- generate sample images for test time (batch norm is different)
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            self.sample_images_test = self.g_forward(
                self.Z, reuse=True, is_training=False
            )

        # ----- discriminator cost (2 steps: for real and fake images) using binary cross-entropy
        # --- (a) for the real images
        self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=tf.ones_like(logits)
        )

        # --- (b) for the fake images
        self.d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=sample_logits,
            labels=tf.zeros_like(sample_logits)
        )

        # --- the total cost consists of the means of the above costs added together
        self.d_cost = tf.reduce_mean(self.d_cost_real) + tf.reduce_mean(self.d_cost_fake)

        # ----- generator cost
        self.g_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=sample_logits,
                labels=tf.ones_like(sample_logits)  # targets set to one
            )
        )

        # --- calculate the accuracy of the discriminator (mostly for debugging)
        real_predictions = tf.cast(logits > 0, tf.float32)
        fake_predictions = tf.cast(sample_logits < 0, tf.float32)
        num_predictions = 2.0 * BATCH_SIZE
        num_correct = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)
        self.d_accuracy = num_correct / num_predictions

        # ----- Setting up the optimizers
        # --- (a) collecting the discriminator and generator parameters
        self.d_params = [t for t in tf.trainable_variables() if t.name.startswith('d')]
        self.g_params = [t for t in tf.trainable_variables() if t.name.startswith('g')]

        # --- (b) discriminator optimizer
        self.d_train_op = tf.train.AdamOptimizer(
            LEARNING_RATE, beta1=BETA1
        ).minimize(
            self.d_cost, var_list=self.d_params
        )

        # --- (c) generator optimizer
        self.g_train_op = tf.train.AdamOptimizer(
            LEARNING_RATE, beta1=BETA1
        ).minimize(
            self.g_cost, var_list=self.g_params
        )

        # --- set up a session and initialize the variables
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

    def build_discriminator(self):
        """
        Creeate all the layers of the discriminator and pass
        the data through to return the logits
        :return: logits (predictions)
        """
        with tf.variable_scope("discriminator") as scope:

            # --- build the convolutional layers
            self.d_convlayers = list()
            mi = self.num_colors
            dim = self.img_dim
            count = 0
            for mo, filter_size, stride, apply_batch_norm in self.d_sizes['conv_layers']:
                name = f"convlayer_{count}"  # name is used for get_variable later
                count += 1
                layer = ConvLayer(name, mi, mo, apply_batch_norm, filter_size, stride, lrelu)
                self.d_convlayers.append(layer)
                mi = mo
                print(f"dim: {dim}")
                # --- keep track of image dimensionality: need this for the first Dense layer
                dim = int(np.ceil(float(dim) / stride))

            # --- get the input dimensionalith for the first Dense layer
            mi = mi * dim * dim

            # --- build the dense layers
            self.d_denselayers = list()
            for mo, apply_batch_norm in self.d_sizes['dense_layers']:
                name = f"denselayer_{count}"
                count += 1
                layer = DenseLayer(name, mi, mo, apply_batch_norm, lrelu)
                mi = mo
                self.d_denselayers.append(layer)

            # --- final logistic regression layer (use it in the d_forward
            #     function below to get the final logits)
            name = f"denselayer_{count}"
            self.d_finallayer = DenseLayer(name, mi, 1, False, lambda x: x)

            # --- get and return the logits
            logits = self.d_forward(self.X)
            return logits

    def d_forward(self, X, reuse=None, is_training=True):
        output = X
        # --- loop throught the created layers and call their forward functions
        for layer in self.d_convlayers:
            output = layer.forward(output, reuse, is_training)
        output = tf.layers.flatten(output)
        for layer in self.d_denselayers:
            output = layer.forward(output, reuse, is_training)
        logits = self.d_finallayer.forward(output, reuse, is_training)
        return logits

    def build_generator(self):
        with tf.variable_scope("generator") as scope:

            # --- calculate image dimensionality at each step (start at the output
            #     of the generator because we know what the size of the image
            #     should be at the end: same as the training data)
            dims = [self.img_dim]
            dim = self.img_dim
            # --- loop backwards through the conv layer sizes and calculate image dims
            for _, _, stride, _ in reversed(self.g_sizes['conv_layers']):
                dim = int(np.ceil(float(dim) / stride))
                dims.append(dim)

            # --- reverse the image dims' list so that it is in the
            #     order of the layers of the generator
            dims = list(reversed(dims))
            print("dims:", dims)
            self.g_dims = dims

            # --- create the first dense layers
            mi = self.latent_dims
            self.g_denselayers = list()
            count = 0
            for mo, apply_batch_norm in self.g_sizes['dense_layers']:
                name = f"g_denselayer_{count}"
                count += 1
                layer = DenseLayer(name, mi, mo, apply_batch_norm)
                self.g_denselayers.append(layer)
                mi = mo

            # --- create the final dense layer (its final size must match the size of the first convolution)
            mo = self.g_sizes['projection'] * dims[0] * dims[0]
            name = f"g_denselayer_{count}"
            layer = DenseLayer(name, mi, mo, not self.g_sizes['bn_after_project'])
            self.g_denselayers.append(layer)

            # --- fractionally shaped convolution layers
            mi = self.g_sizes['projection']
            self.g_convlayers = list()

            # --- output may use tanh or sigmoid
            num_relus = len(self.g_sizes['conv_layers']) - 1
            activation_functions = [tf.nn.relu] * num_relus + [self.g_sizes['output_activation']]

            for i in range(len(self.g_sizes['conv_layers'])):
                name = "fs_convlayer_%s" % i
                mo, filtersz, stride, apply_batch_norm = self.g_sizes['conv_layers'][i]
                f = activation_functions[i]
                output_shape = [self.batch_size, dims[i + 1], dims[i + 1], mo]
                print("mi:", mi, "mo:", mo, "outp shape:", output_shape)
                layer = FractionallyStridedConvLayer(
                    name, mi, mo, output_shape, apply_batch_norm, filtersz, stride, f
                )
                self.g_convlayers.append(layer)
                mi = mo

            # get the output
            self.g_sizes = self.g_sizes
            return self.g_forward(self.Z)

    def g_forward(self, Z, reuse=None, is_training=True):
        # --- dense layers
        output = Z
        for layer in self.g_denselayers:
            output = layer.forward(output, reuse, is_training)

        # --- reshape the flat vector into an image
        output = tf.reshape(
            output,
            [-1, self.g_dims[0], self.g_dims[0], self.g_sizes['projection']],
        )

        # --- batch norm
        with tf.variable_scope('bn_after_project') as scope:
            if self.g_sizes['bn_after_project']:
                output = tf.layers.batch_normalization(
                    output,
                    momentum=0.9,
                    epsilon=1e-5,
                    scale=True,
                    training=is_training,
                    reuse=reuse,
                )

        # --- pass through fractionally strided conv layers
        for layer in self.g_convlayers:
            output = layer.forward(output, reuse, is_training)

        return output

    def fit(self, X):
        d_costs = list()
        g_costs = list()

        N = len(X)
        n_batches = N // BATCH_SIZE
        total_iters = 0
        for i in range(EPOCHS):
            print("epoch:", i)
            np.random.shuffle(X)
            for j in range(n_batches):
                t0 = datetime.now()

                if type(X[0]) is str:
                    # --- celeb dataset
                    batch = utils.files2images(
                        X[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                    )

                else:
                    # --- mnist dataset
                    batch = X[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]

                Z = np.random.uniform(-1, 1, size=(BATCH_SIZE, self.latent_dims))

                # --- train the discriminator
                _, d_cost, d_acc = self.sess.run(
                    (self.d_train_op, self.d_cost, self.d_accuracy),
                    feed_dict={self.X: batch, self.Z: Z, self.batch_size: BATCH_SIZE},
                )
                d_costs.append(d_cost)

                # --- train the generator (twice!)
                _, g_cost1 = self.sess.run(
                    (self.g_train_op, self.g_cost),
                    feed_dict={self.Z: Z, self.batch_size: BATCH_SIZE},
                )
                _, g_cost2 = self.sess.run(
                    (self.g_train_op, self.g_cost),
                    feed_dict={self.Z: Z, self.batch_size: BATCH_SIZE},
                )
                g_costs.append((g_cost1 + g_cost2) / 2)  # just use the avg

                print(f"batch: {j + 1}/{n_batches}  -  dt: {datetime.now() - t0} - d_acc: {d_acc:.2f}")

                # --- save samples periodically
                total_iters += 1
                if total_iters % SAVE_SAMPLE_PERIOD == 0:
                    print("saving a sample...")
                    samples = self.sample(64)  # shape is (64, D, D, color)

                    # for convenience
                    d = self.img_dim

                    if samples.shape[-1] == 1:
                        # --- if color == 1, want a 2-D image (N x N)
                        samples = samples.reshape(64, d, d)
                        flat_image = np.empty((8 * d, 8 * d))

                        k = 0
                        for i in range(8):
                            for j in range(8):
                                flat_image[i * d:(i + 1) * d, j * d:(j + 1) * d] = samples[k].reshape(d, d)
                                k += 1
                    else:
                        # --- if color == 3, want a 3-D image (N x N x 3)
                        flat_image = np.empty((8 * d, 8 * d, 3))
                        k = 0
                        for i in range(8):
                            for j in range(8):
                                flat_image[i * d:(i + 1) * d, j * d:(j + 1) * d] = samples[k]
                                k += 1
                        plt.imshow(flat_image)
                    imageio.imwrite(f"samples/samples_at_iter_{total_iters}.png", flat_image)

        # --- save a plot of the costs
        plt.clf()
        plt.plot(d_costs, label='discriminator cost')
        plt.plot(g_costs, label='generator cost')
        plt.legend()
        plt.savefig('cost_vs_iteration.png')

    def sample(self, n):
        Z = np.random.uniform(-1, 1, size=(n, self.latent_dims))
        samples = self.sess.run(self.sample_images_test, feed_dict={self.Z: Z, self.batch_size: n})
        return samples


def fit_to_celeb():
    X = utils.get_celeb()
    dim = 108
    colors = 3

    # --- decoder setup
    d_sizes = {
        'conv_layers': [
            (dim, 5, 2, False),
            (128, 5, 2, True),
            (256, 5, 2, True),
            (512, 5, 2, True)
        ],
        'dense_layers': [],
    }

    # --- generator setup
    g_sizes = {
        'z': 100,
        'projection': 512,
        'bn_after_project': True,
        'conv_layers': [
            (256, 5, 2, True),
            (128, 5, 2, True),
            (dim, 5, 2, True),
            (colors, 5, 2, False)
        ],
        'dense_layers': [],
        'output_activation': tf.tanh,  # since the data are normalized to be between -1 and +1
    }

    gan = DCGAN(dim, colors, d_sizes, g_sizes)
    gan.fit(X)


def fit_to_mnist():
    X, Y = utils.get_mnist()
    X = X.reshape(len(X), 28, 28, 1)
    dim = X.shape[1]
    colors = X.shape[-1]

    # --- decoder setup
    d_sizes = {
        'conv_layers': [(2, 5, 2, False), (64, 5, 2, True)],
        'dense_layers': [(1024, True)],
    }

    # --- generator setup
    g_sizes = {
        'z': 100,
        'projection': 128,
        'bn_after_project': False,
        'conv_layers': [(128, 5, 2, True), (colors, 5, 2, False)],
        'dense_layers': [(1024, True)],
        'output_activation': tf.sigmoid,
    }

    # --- DCGAN setup (assume square images, so only need 1 dim)
    gan = DCGAN(dim, colors, d_sizes, g_sizes)
    gan.fit(X)


if __name__ == '__main__':
    fit_to_celeb()
    # fit_to_mnist()
