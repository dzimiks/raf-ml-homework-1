import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def create_feature_matrix(x, nb_features):
	"""
	An auxiliary function that creates feature matrix (M x N) from a series of training examples.

	:param x: Array of numbers.
	:param nb_features: Degree of a polynomial.
	:return: The array formed by stacking the given arrays.
	"""

	tmp_features = []

	for deg in range(1, nb_features + 1):
		tmp_features.append(np.power(x, deg))

	return np.column_stack(tmp_features)


tf.reset_default_graph()

# Avoid a scientific notation and round off on 5 decimals
np.set_printoptions(suppress=True, precision=5)

# Step 1: Loading and processing data
filename = 'data/funky.csv'
all_data = np.loadtxt(filename, delimiter=',')
data = dict()
data['x'] = all_data[:, 0]
data['y'] = all_data[:, 1]

# Random shuffle
nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

# Normalization
data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

# Feature matrix creation
# This variable controls the number of features, ie. degree of polynomial
nb_features = 6
print('Original values (first 3):')
print(data['x'][:3])
print('Feature matrix (first 3 rows):')
data['x'] = create_feature_matrix(data['x'], nb_features)
print(data['x'][:3, :])

# Plotting all data
# plt.scatter(data['x'][:, 0], data['y'])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# Step 2: Model.
X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
Y = tf.placeholder(shape=None, dtype=tf.float32)
w = tf.Variable(tf.zeros(nb_features))
bias = tf.Variable(0.0)

w_col = tf.reshape(w, (nb_features, 1))
hyp = tf.add(tf.matmul(X, w_col), bias)

# Step 3: Cost function and optimization
Y_col = tf.reshape(Y, (-1, 1))
loss = tf.reduce_mean(tf.square(hyp - Y_col))

# Switching to AdamOptimizer because the GradientDescent loses with more complex functions
opt_op = tf.train.AdamOptimizer().minimize(loss)

# Step 4: Training
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# 1000 epochs of training
	nb_epochs = 1000

	for epoch in range(nb_epochs):
		# Stochastic Gradient Descent.
		epoch_loss = 0

		for sample in range(nb_samples):
			feed = {X: data['x'][sample].reshape((1, nb_features)),
					Y: data['y'][sample]}
			_, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
			epoch_loss += curr_loss

			if epoch == nb_epochs - 1 and data['x'][sample][0] == 3:
				print(data['x'][sample])
				print(curr_loss)

		# Writes average loss in every hundred epoch
		epoch_loss /= nb_samples

		if (epoch + 1) % 100 == 0:
			print('Epoch: {}/{} | Avg loss: {:.5f}'.format(epoch + 1, nb_epochs, epoch_loss))

	# Print and plot the final value of the parameters
	w_val = sess.run(w)
	bias_val = sess.run(bias)
	print('w = ', w_val, 'bias = ', bias_val)
	xs = create_feature_matrix(np.linspace(-2, 4, 100), nb_features)
	hyp_val = sess.run(hyp, feed_dict={X: xs})
	plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color='g')
	plt.scatter(data['x'][:, 0], data['y'])
	plt.xlim([-2, 2])
	plt.ylim([-3, 3.5])
	plt.title('Features ' + str(nb_features))
	plt.show()
