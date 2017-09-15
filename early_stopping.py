# early stopping regularization for neural networks

#import tensorflow as tf
import numpy as np 
import pandas as pd 
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split


RANDOM_STATE = 42
# FIG_SIZE = (10, 7)


features, target = load_digits(return_X_y=True)

# Make a train/test split using 20% test size
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.20,
                                                    random_state=RANDOM_STATE)




import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



print mnist.train.images[0]


# x and y are placeholders for our training data
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))# Our model of y = a*x + b
y = tf.nn.softmax(tf.matmul(x, W) + b)

# placeholder for the true labels
y_ = tf.placeholder(tf.float32, [None, 10])

# Our error is defined as the cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()

errors = []

# this is the main "hyperparameter" associated with early stopping
patience = 40
p = 0
i = 0
bestValAcc = -np.inf

with tf.Session() as session:
    session.run(model)
    while p < patience:
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, error_value = session.run([train_op, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        errors.append(error_value)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = session.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
#         print acc
        if acc > bestValAcc:
            p = 0
            bestValAcc = acc
        else:
            p += 1
        i += 1
    
    print "p:",p,"accuracy:", session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print "iterations:", i


# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
# plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
# plt.show()



