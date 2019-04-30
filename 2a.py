import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix(x, nb_features):
	tmp_features = []
	for deg in range(1, nb_features + 1):
		tmp_features.append(np.power(x, deg))
	return np.column_stack(tmp_features)


def polynomial_regression(input_data, nb_samples, nb_features, color):
	# Restartuj graf
	tf.reset_default_graph()

	# Kreiranje feature matrice.
	data = input_data.copy()
	data['x'] = create_feature_matrix(data['x'], nb_features)

	# Korak 2: Model.
	X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
	Y = tf.placeholder(shape=None, dtype=tf.float32)
	w = tf.Variable(tf.zeros(nb_features))
	bias = tf.Variable(0.0)

	w_col = tf.reshape(w, (nb_features, 1))
	hyp = tf.add(tf.matmul(X, w_col), bias)

	# Korak 3: Funkcija troška i optimizacija.
	Y_col = tf.reshape(Y, (-1, 1))
	loss = tf.reduce_mean(tf.square(hyp - Y_col))

	# Prelazimo na AdamOptimizer jer se prost GradientDescent lose snalazi sa
	# slozenijim funkcijama.
	opt_op = tf.train.AdamOptimizer().minimize(loss)

	# Korak 4: Trening.
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Izvršavamo 200 epoha treninga.
		nb_epochs = 200
		for epoch in range(nb_epochs):

			# Stochastic Gradient Descent.
			epoch_loss = 0
			for sample in range(nb_samples):
				feed = {X: data['x'][sample].reshape((1, nb_features)),
						Y: data['y'][sample]}
				_, curr_loss = sess.run([opt_op, loss], feed_dict=feed)
				epoch_loss += curr_loss

			# U svakoj desetoj epohi ispisujemo prosečan loss.
			epoch_loss /= nb_samples
			if (epoch + 1) % 10 == 0:
				print('Epoch: {}/{} | Avg loss: {:.5f}'.format(epoch + 1, nb_epochs, epoch_loss))

		# Ispisujemo i plotujemo finalnu vrednost parametara.
		w_val = sess.run(w)
		bias_val = sess.run(bias)
		print('w = ', w_val, 'bias = ', bias_val)
		xs = create_feature_matrix(np.linspace(-2, 4, 100), nb_features)
		hyp_val = sess.run(hyp, feed_dict={X: xs})  # Bez Y jer nije potrebno.
		plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color=color)
		plt.xlim([-2, 2])
		plt.ylim([-3, 4])

		return sess.run(loss, feed_dict={X: data['x'], Y: data['y']})


def main():
	# Izbegavamo scientific notaciju i zaokruzujemo na 5 decimala.
	np.set_printoptions(suppress=True, precision=5)

	# Korak 1: Učitavanje i obrada podataka.
	filename = 'data/funky.csv'
	all_data = np.loadtxt(filename, delimiter=',', skiprows=0, usecols=(0, 1))
	data = dict()
	data['x'] = all_data[:, 0]
	data['y'] = all_data[:, 1]

	# Nasumično mešanje.
	nb_samples = data['x'].shape[0]
	indices = np.random.permutation(nb_samples)
	data['x'] = data['x'][indices]
	data['y'] = data['y'][indices]

	# Normalizacija (obratiti pažnju na axis=0).
	data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
	data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

	# Iscrtavanje podataka
	plt.scatter(data['x'], data['y'])
	plt.xlabel('X value')
	plt.ylabel('Y value')

	colors = ['g', 'm', 'b', 'r', 'k', 'y']
	losses = []

	for degree in range(1, 7):
		print('Polynomial regression for degree = ', degree)
		loss = polynomial_regression(data, nb_samples, degree, colors[degree - 1])
		losses.append(loss)
		print('loss = {:.5f}'.format(loss))
		print('----------------------------')

	plt.show()

	plt.plot(range(1, 7), losses, color='g')
	plt.xlabel('Degree')
	plt.ylabel('Loss')
	plt.show()


if __name__ == '__main__':
	main()


# Mozemo primetiti da polinomi stepena 3, 4, 5 i 6 dobro aproksimiraju skup podataka i imaju loss blizu 0.1.
# Polinom stepena 3 ima najmanji loss (jedini ispod 0.11).

# Polinomi stepena 1 i 2 su losija aproksimacija i imaju loss oko 0.3
# Polinom stepena 1 definise pravu koja nije dobra aproksimacija podataka.
