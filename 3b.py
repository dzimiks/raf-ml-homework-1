# 3b
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

		self.X = tf.placeholder(shape=(None, numOfFeatures), dtype=tf.float32)
		self.Y = tf.placeholder(shape=None, dtype=tf.int32)
		self.T = tf.placeholder(shape=numOfFeatures, dtype=tf.float32)

		dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.T)), axis=1))
		_, idxs = tf.nn.top_k(-dists, self.k)

		self.classes = tf.gather(self.Y, idxs)
		self.dists = tf.gather(dists, idxs)

		self.w = tf.fill([coeff], 1 / coeff)

		# Svaki red mnozimo svojim glasom i sabiramo glasove po kolonama.
		w_col = tf.reshape(self.w, (coeff, 1))
		self.classes_one_hot = tf.one_hot(self.classes, numOfClasses)
		self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis=0)

		# Klasa sa najvise glasova je hipoteza.
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
				if test_data['y'] is not None:
					actual = test_data['y'][i]
					match = (int(hyp_val) == int(actual))
					if match:
						matches += 1

			accuracy = matches / numOfQueries
			return accuracy


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

	# ispis rezultata
	numOfClasses = 3

	different_k = []
	accuracy_res = []
	average_accuracy = 0

	for i in range(1, 16):
		coeff = i
		knn = KNN(numOfFeatures, numOfClasses, train_data, coeff)
		accuracy = knn.predict(test_data)
		different_k.append(i)
		accuracy_res.append(accuracy)
		average_accuracy += accuracy

	average_accuracy /= 15
	print("Average accuracy: " + str(average_accuracy))
	fig, drawing = plt.subplots()

	drawing.plot(different_k, accuracy_res, color='red')
	drawing.set(xlabel='Value k in KNN', ylabel='Accuracy', title='Dependence Accuracy of value k')

	drawing.grid()
	plt.show()

# Na osnovu više pokretanja programa i iscrtavanja grafika, mogu se zaključiti sledeće stvari:
# - Accuracy varira od [0.6, 0.9] na većini grafika 
# - Prosečan Accuracy takođe varira [0.7, 0.85]
# - Generalno najbolji rezultati se dobijaju za vrednosti k>=5, uglavno je pik za k u intervalu [8, 12]
# - Ne može se primetiti neko značajno pravilo, nakon početnog rasta, Accuracy se prilično proizvoljno ponaša za veće vrednosti k 
# - Mnogo stvari zavisi od toga kako su promešani podaci
# - Na osnovu Accuracy za vrednost k = 1, možemo prilično pouzdano da procenimo kako će izgledati Accuracy u ostalim tačkama:
#     - uglavnom kada je bio manji Accuracy za k = 1, negde oko 0.6, maximalni Accuracy se isto nije kretao preko 0.8 
#     - slično kada je Accuracy > 0.7 ya k = 1, uglavnom je pik bio preko 0.85
