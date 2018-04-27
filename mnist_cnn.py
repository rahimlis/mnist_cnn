from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# Application code will go here

def cnn_model_fn(features, labels, mode):
    """
    Model function for CNN

 INPUT   ->  CONV2D (5x5x32 f) ->   POOL   -> CONV2D (5x5x64 f)  -> POOL   ->  FLATTEN        ->    FC
28x28x1        28x28x32           14x14x32       14x14x64          7x7x64    batch_size x(7*7*64)  10x1024


    """

    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding='same',
                             name='conv1',
                             activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,name='pool1')

    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding='same',
                             name='conv2',
                             activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,name='pool2')

    # now we unfold pool2 layer into dense (FC) layer

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64],name='pool2_flat')

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,name='dropout')

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN,name='dropout')

    logits = tf.layers.dense(inputs=dropout, units=10,name='logits')

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (both TRAIN and EVAL modes)


    print("shape of labels: "+str(labels.shape))
    print("shape of logits: "+str(logits.shape))

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Training mode configuration
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Evaluation metrics


    eval_metrics_ops = {
        "accuracy": tf.metrics.accuracy(labels=tf.argmax(labels, 0), predictions=predictions['classes'])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)



def main(unused_argv):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data = mnist.train.images
    tf.summary.image('input',train_data)
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tmp/mnist_model")

    #tensors_to_log = {"softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(tensors=None, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                        y=train_labels,
                                                        batch_size=100,
                                                        num_epochs=None,
                                                        shuffle=True)

    mnist_classifier.train(input_fn=train_input_fn, steps=15000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                       y=eval_labels,
                                                       batch_size=10,
                                                       num_epochs=1,
                                                       shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)



if __name__ == '__main__':
    tf.app.run()

