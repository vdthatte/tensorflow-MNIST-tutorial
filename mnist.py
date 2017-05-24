from tensorflow.examples.tutorials.mnist import input_data 								# import data
import tensorflow as tensorflow															# import library

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)							# save datasets in a variable
x = tf.placeholder(tf.float32, [None, 784])												# 2d tensor of floating point number with a shape [None, 784]
W = tf.Variable(tf.zeros([784, 10]))													# multiply the 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence
b = tf.Variable(tf.zeros([10]))															# b has a shape [10] and we can add this to the output
y = tf.nn.softmax(tf.matmul(x, W) + b)													# matrix multiply, x and W and add b and perform softmax present in tensorflows neural networks
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))	# use cross entropy to determine the loss of a machine learning model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)				# train using the back propagation algorithm, gradient descent
sess = tf.InteractiveSession()															# launch the model in interative session
tf.global_variables_initializer().run()													# create an operation
for _ in range(1000):																	# run the training step a thousand times
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))							# get the correct/wrong predictions in an array

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))						# calculate the accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))		# print the accuracy		

