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
		self.k = coeff;

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


# Učitavanje podataka iz csv file-a

filename = 'data/iris.csv'
data = dict()
numOfFeatures = 4
data['x'] = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(range(0, numOfFeatures)))
data['y'] = np.loadtxt(filename, dtype='str', delimiter=',', skiprows=1, usecols=4)

for i in range(0, len(data['y'])):
	if data['y'][i] == 'Iris-setosa':
		data['y'][i] = 0
	elif data['y'][i] == 'Iris-versicolor':
		data['y'][i] = 1
	else:
		data['y'][i] = 2

# crtanje incijalnog grafika i ispis podataka

# print(len(data['x']))
# print(len(data['y']))
# print(data['x'])
# print(data['y'])
# plt.xlim(8)
# plt.xlabel('sepal_length')
# plt.ylim(5)
# plt.ylabel('sepal_width')
# plt.scatter(data['x'][:,0], data['x'][:,1], edgecolors='r')

# Nasumično mešanje.

nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

# print(data['y'])

# deljenje podataka za trening i test po TRAIN_CONST

train_data = dict()
test_data = dict()

totalDataLen = len(data['x'])
totalTrainLen = int((totalDataLen * TRAIN_CONST) / 100);
train_data['x'] = data['x'][0:totalTrainLen]
train_data['y'] = data['y'][0:totalTrainLen]
test_data['x'] = data['x'][totalTrainLen:totalDataLen]
test_data['y'] = data['y'][totalTrainLen:totalDataLen]

# plt.scatter(train_data['x'][:,0], train_data['x'][:,1], edgecolors = 'b')

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

# Grafik se slično ponaša i kada su uključena sva 4 feature-a kao u slučaju kada su uključena 2 - ne postoji neka optimalna vrednost za k, više se random kreće Accuracy
# Ključna razlika je u tome što se Accuracy značajno povećao i uvek se kreće u opsegu 0.88 - 1.00. 
# Prosečni Accuracy se povećao za otprilike 0.2 i kreće se u opsegu [0.93, 0.98]
# Sve to dovodi do zaključka da je bolje uzeti više feature-a i sve ih koristiti u KNN-u, jer će rezultati biti tačniji i greške su skoro zanemarljive
# Jedina mana ovog pristupa može biti u tome što se povećava vreme izvršavanja programa.
