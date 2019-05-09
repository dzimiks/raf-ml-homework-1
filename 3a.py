# 3a
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.reset_default_graph()

TRAIN_CONST = 80  # procentualno koliko podataka koristimo za trening


# klasa za KNN
class KNN:
	def __init__(self, numOfFeatures, numOfClasses, train_data, coeff):
		self.numOfFeatures = numOfFeatures
		self.numOfClasses = numOfClasses
		self.data = train_data
		self.k = coeff

		# Gradimo model, X je matrica podataka a Q je vektor koji predstavlja upit.
		self.X = tf.placeholder(shape=(None, numOfFeatures), dtype=tf.float32)
		self.Y = tf.placeholder(shape=None, dtype=tf.int32)
		self.T = tf.placeholder(shape=numOfFeatures, dtype=tf.float32)
		self.predRes = []

		# Racunamo kvadriranu euklidsku udaljenost i uzimamo minimalnih k.
		dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.T)), axis=1))
		_, idxs = tf.nn.top_k(-dists, self.k)

		self.classes = tf.gather(self.Y, idxs)
		self.dists = tf.gather(dists, idxs)

		self.w = tf.fill([coeff], 1 / coeff)

		# Svaki red mnozimo svojim glasom i sabiramo glasove po kolonama.
		w_col = tf.reshape(self.w, (coeff, 1))
		self.classes_one_hot = tf.one_hot(self.classes, numOfClasses)
		self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis=0)

		# Klasa sa najviše glasova je hipoteza.
		self.hyp = tf.argmax(self.scores)

	def predict(self, test_data):
		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())
			numOfQueries = test_data['x'].shape[0]
			matches = 0

			for i in range(numOfQueries):
				hyp_val = sess.run(self.hyp, feed_dict={self.X: self.data['x'],
														self.Y: self.data['y'],
														self.T: test_data['x'][i]})
				self.predRes.append(hyp_val)
				if test_data['y'] is not None:
					actual = test_data['y'][i]
					match = (int(hyp_val) == int(actual))
					if match:
						matches += 1

			accuracy = matches / numOfQueries
			return accuracy

	def predictions(self, test_data):

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			numOfQueries = len(test_data)

			for i in range(numOfQueries):
				hyp_val = sess.run(self.hyp, feed_dict={self.X: self.data['x'],
														self.Y: self.data['y'],
														self.T: test_data[i]})
				self.predRes.append(hyp_val)

		return np.asarray(self.predRes)


if __name__ == '__main__':
	# Učitavanje podataka iz csv file-a
	filename = 'data/iris.csv'
	data = dict()
	numOfFeatures = 2
	data['x'] = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(range(0, numOfFeatures)))
	data['y'] = np.loadtxt(filename, dtype='str', delimiter=',', skiprows=1, usecols=4)

	for i in range(0, len(data['y'])):
		if data['y'][i] == 'Iris-setosa':
			data['y'][i] = 0
		elif data['y'][i] == 'Iris-versicolor':
			data['y'][i] = 1
		else:
			data['y'][i] = 2

	# Nasumično mešanje.
	nb_samples = data['x'].shape[0]
	indices = np.random.permutation(nb_samples)
	data['x'] = data['x'][indices]
	data['y'] = data['y'][indices]

	# deljenje podataka za trening i test po TRAIN_CONST
	train_data = dict()
	test_data = dict()

	totalDataLen = len(data['x'])
	totalTrainLen = int((totalDataLen * TRAIN_CONST) / 100)
	train_data['x'] = data['x'][0:totalTrainLen]
	train_data['y'] = data['y'][0:totalTrainLen]
	test_data['x'] = data['x'][totalTrainLen:totalDataLen]
	test_data['y'] = data['y'][totalTrainLen:totalDataLen]

	# plt.scatter(train_data['x'][:,0], train_data['x'][:,1], edgecolors = 'b')

	# ispis rezultata
	numOfClasses = 3
	coeff = 3
	knn = KNN(numOfFeatures, numOfClasses, train_data, coeff)
	accuracy = knn.predict(test_data)

	print('Test set accuracy: ', accuracy)

	# Generisemo grid.
	step_size = 0.02
	x1, x2 = np.meshgrid(np.arange(min(train_data['x'][:, 0]) - 0.3, max(train_data['x'][:, 0]) + 0.3,
								   step_size),
						 np.arange(min(train_data['x'][:, 1]) - 0.3, max(train_data['x'][:, 1]) + 0.3,
								   step_size))
	x_feed = np.vstack((x1.flatten(), x2.flatten())).T

	# Racunamo vrednost hipoteze.
	knn = KNN(numOfFeatures, numOfClasses, train_data, coeff)

	pred_val = knn.predictions(x_feed)
	pred_plot = pred_val.reshape([x1.shape[0], x1.shape[1]])

	# Crtamo contour plot.
	from matplotlib.colors import LinearSegmentedColormap

	classes_cmap = LinearSegmentedColormap.from_list('classes_cmap',
													 ['lightblue',
													  'lightgreen',
													  'tomato'])
	plt.contourf(x1, x2, pred_plot, cmap=classes_cmap, alpha=0.7)

	# Crtamo sve podatke preko.
	idxs_0 = []

	for i in range(0, len(train_data['y'])):
		if int(train_data['y'][i]) == 0:
			idxs_0.append(i)

	idxs_1 = []

	for i in range(0, len(train_data['y'])):
		if int(train_data['y'][i]) == 1:
			idxs_1.append(i)

	idxs_2 = []

	for i in range(0, len(train_data['y'])):
		if int(train_data['y'][i]) == 2:
			idxs_2.append(i)

	plt.scatter(train_data['x'][idxs_0, 0], train_data['x'][idxs_0, 1], c='b',
				edgecolors='k', label='Iris-setosa')
	plt.scatter(train_data['x'][idxs_1, 0], train_data['x'][idxs_1, 1], c='g',
				edgecolors='k', label='Iris-versicolor')
	plt.scatter(train_data['x'][idxs_2, 0], train_data['x'][idxs_2, 1], c='r',
				edgecolors='k', label='Iris-virginica')
	plt.legend()
	plt.show()
