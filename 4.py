from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import numpy as np
import math
import os

VOCAB_SIZE = 500


class MultinomialNaiveBayes:
	def __init__(self, nb_classes, nb_words, pseudocount):
		self.nb_classes = nb_classes
		self.nb_words = nb_words
		self.pseudocount = pseudocount

	def fit(self, X, Y):
		nb_examples = X.shape[0]

		# Racunamo P(Klasa) - priors
		# np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
		# broja u intervalu [0, maksimalni broj u listi]
		self.priors = np.bincount(Y) / nb_examples
		print('Priors:')
		print(self.priors)

		# Racunamo broj pojavljivanja svake reci u svakoj klasi
		occs = np.zeros((self.nb_classes, self.nb_words))
		for i in range(nb_examples):
			c = Y[i]
			for w in range(self.nb_words):
				cnt = X[i][w]
				occs[c][w] += cnt
		print('Occurences:')
		print(occs)

		# Racunamo P(Rec_i|Klasa) - likelihoods
		self.like = np.zeros((self.nb_classes, self.nb_words))
		for c in range(self.nb_classes):
			for w in range(self.nb_words):
				up = occs[c][w] + self.pseudocount
				down = np.sum(occs[c]) + self.nb_words * self.pseudocount
				self.like[c][w] = up / down
		print('Likelihoods:')
		print(self.like)

	def predict(self, bow):
		# Racunamo P(Klasa|bow) za svaku klasu
		probs = np.zeros(self.nb_classes)
		for c in range(self.nb_classes):
			prob = np.log(self.priors[c])
			for w in range(self.nb_words):
				cnt = bow[w]
				prob += cnt * np.log(self.like[c][w])
			probs[c] = prob
		# Trazimo klasu sa najvecom verovatnocom
		# print('\"Probabilites\" for a test BoW (with log):')
		# print(probs)
		prediction = np.argmax(probs)
		return prediction

	def predict_multiply(self, bow):
		# Racunamo P(Klasa|bow) za svaku klasu
		# Mnozimo i stepenujemo kako bismo uporedili rezultate sa slajdovima
		probs = np.zeros(self.nb_classes)
		for c in range(self.nb_classes):
			prob = self.priors[c]
			for w in range(self.nb_words):
				cnt = bow[w]
				prob *= self.like[c][w] ** cnt
			probs[c] = prob
		# Trazimo klasu sa najvecom verovatnocom
		# print('\"Probabilities\" for a test BoW (without log):')
		# print(probs)
		prediction = np.argmax(probs)
		return prediction


def occ_score(word, doc):
	return 1 if word in doc else 0


def numocc_score(word, doc):
	return doc.count(word)


def freq_score(word, doc):
	return doc.count(word) / len(doc)


def get_clean_doc(doc):
	porter = PorterStemmer()
	stop_punc = set(stopwords.words('english')).union(set(punctuation)).union({'br'})
	table = str.maketrans('', '', punctuation)
	words = wordpunct_tokenize(doc)
	words_lower = [w.lower() for w in words]
	words_stripped = [w.translate(table) for w in words_lower]  # izbaci sve znakove iz rijeci
	words_filtered = [w for w in words_stripped if w not in stop_punc and w.isalpha()]
	words_stemmed = [porter.stem(w) for w in words_filtered]
	return words_stemmed


def create_vocab(corpus):
	# Kreiramo vokabular
	print('Creating the vocab...')
	vocab_dict = dict()
	for doc in corpus:
		for word in doc:
			vocab_dict.setdefault(word, 0)
			vocab_dict[word] += 1

	return sorted(vocab_dict, key=vocab_dict.get, reverse=True)[:VOCAB_SIZE]


def create_feature_matrix(corpus, labels, vocab):
	# 1: Bag of Words model
	print('Creating BOW features...')
	X = np.zeros((len(corpus), len(vocab)), dtype=np.float64)
	for doc_idx in range(len(corpus)):
		doc = corpus[doc_idx]
		X[doc_idx] = create_bow(doc, vocab)

	Y = np.zeros(len(corpus), dtype=np.int32)
	for i in range(len(Y)):
		Y[i] = labels[i]

	return X, Y


def clean_data(corpus):
	# Priprema podataka
	porter = PorterStemmer()

	# Cistimo korpus
	print('Cleaning the corpus...')
	clean_corpus = []
	for doc in corpus:
		clean_corpus.append(get_clean_doc(doc))

	return clean_corpus


def create_bow(doc, vocab):
	bow = np.zeros(len(vocab), dtype=np.float64)
	for word_idx in range(len(vocab)):
		word = vocab[word_idx]
		cnt = numocc_score(word, doc)
		bow[word_idx] = cnt
	return bow


def read_data():
	corpus = []
	labels = []
	print('Reading data...')
	for root, _, files in os.walk('./data/imdb/'):
		for name in files:
			file_path = os.path.join(root, name)
			with open(file_path, 'r', encoding='utf8') as f:
				file_data = f.read()
				corpus.append(file_data)

			label = 1 if 'pos' in file_path else 0
			labels.append(label)

	indices = np.random.permutation(len(corpus))

	corpus = np.asarray(corpus)
	corpus = corpus[indices]

	labels = np.asarray(labels)
	labels = labels[indices]

	return corpus, labels


def get_top_k(corpus, labels, k, label):
	cnt = 0
	count_dict = dict()
	for i in range(len(corpus)):
		if labels[i] == label:
			cnt += 1
			for word in corpus[i]:
				count_dict.setdefault(word, 0)
				count_dict[word] += 1

	print(cnt)
	return sorted(count_dict, key=count_dict.get, reverse=True)[:k]


def main():
	np.set_printoptions(precision=12, linewidth=200)

	corpus, labels = read_data()
	print('corpus len = ', len(corpus))

	limit = math.floor(len(corpus) * 0.80)
	train_corpus = corpus[:limit]
	train_labels = labels[:limit]

	test_corpus = corpus[limit:]
	test_corpus = clean_data(test_corpus)

	test_labels = labels[limit:]

	print(len(train_corpus), len(train_labels))
	print(len(test_corpus), len(test_labels))

	train_corpus = clean_data(train_corpus)
	vocab = create_vocab(train_corpus)
	print(vocab)
	X, Y = create_feature_matrix(train_corpus, train_labels, vocab)

	print()
	print(X)
	print(Y)
	print()

	class_names = ['Negative', 'Positive']
	model = MultinomialNaiveBayes(nb_classes=2, nb_words=VOCAB_SIZE, pseudocount=1)
	model.fit(X, Y)

	correct_pred = 0
	TP = 0
	FP = 0
	TN = 0
	FN = 0

	print()
	print('Running tests...')
	for i in range(len(test_corpus)):
		doc = test_corpus[i]
		label = test_labels[i]
		bow = create_bow(doc, vocab)

		prediction = model.predict(bow)
		# print('Predicted class (with log): ', class_names[prediction], class_names[test_labels[i]])

		if prediction == test_labels[i]:
			correct_pred += 1

		if prediction == 1:
			if test_labels[i] == 1:
				TP += 1
			else:
				FP += 1
		else:
			if test_labels[i] == 0:
				TN += 1
			else:
				FN += 1

	acc = correct_pred / len(test_corpus)
	print(TP + FN + FP + TN)
	conf_mat = [[TN, FP], [FN, TP]]

	print('Accuracy = {}'.format(acc))
	print('Confussion matrix', conf_mat)

	print('Top 5 in negative', get_top_k(train_corpus, train_labels, 5, 0))
	print('Top 5 in postiive', get_top_k(train_corpus, train_labels, 5, 1))


if __name__ == '__main__':
	main()
