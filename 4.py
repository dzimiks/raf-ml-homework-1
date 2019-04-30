from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import math
import os

VOCAB_SIZE = 10000


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


def clean_data(corpus, labels):
	# Priprema podataka
	porter = PorterStemmer()

	# Cistimo korpus
	print('Cleaning the corpus...')
	clean_corpus = []
	for doc in corpus:
		clean_corpus.append(get_clean_doc(doc))

	# Kreiramo vokabular
	print('Creating the vocab...')
	vocab_dict = dict()
	for doc in clean_corpus:
		for word in doc:
			vocab_dict.setdefault(word, 0)
			vocab_dict[word] += 1

	vocab = sorted(vocab_dict, key=vocab_dict.get, reverse=True)[:VOCAB_SIZE]
	# for word in vocab:
	#     print(word, vocab_dict[word])

	# print('Vocab:', list(zip(vocab, range(len(vocab)))))
	print('Feature vector size: ', len(vocab))

	# 1: Bag of Words model
	print('Creating BOW features...')
	X = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float64)
	for doc_idx in range(len(clean_corpus)):
		doc = clean_corpus[doc_idx]
		X[doc_idx] = create_bow(doc, vocab)

	Y = np.zeros(len(clean_corpus), dtype=np.int32)
	for i in range(len(Y)):
		Y[i] = labels[i]

	return X, Y, vocab


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


def main():
	np.set_printoptions(precision=12, linewidth=200)

	corpus, labels = read_data()
	print('corpus len = ', len(corpus))

	limit = math.floor(len(corpus) * 0.80)
	train_corpus = corpus[:limit]
	train_labels = labels[:limit]

	test_corpus = corpus[limit:]
	test_labels = labels[limit:]

	print(len(train_corpus), len(train_labels))
	print(len(test_corpus), len(test_labels))

	X, Y, vocab = clean_data(train_corpus, train_labels)
	# print(vocab)
	print()
	print(X)
	print(Y)
	print(len(Y))
	print()

	class_names = ['Negative', 'Positive']
	model = MultinomialNaiveBayes(nb_classes=2, nb_words=VOCAB_SIZE, pseudocount=1)
	model.fit(X, Y)

	success = 0
	print()
	for i in range(len(test_corpus)):
		doc = get_clean_doc(test_corpus[i])
		label = test_labels[i]
		bow = create_bow(doc, vocab)
		# print()
		# print(bow)
		# bow = np.zeros(VOCAB_SIZE)

		prediction = model.predict(bow)
		# print(test_corpus[i])
		# print('Predicted class (with log): ', class_names[prediction], class_names[test_labels[i]])

		if prediction == test_labels[i]:
			success += 1
	# prediction = model.predict_multiply(test_bow)
	# print('Predicted class (without log): ', class_names[prediction])

	print(success / len(test_corpus))


if __name__ == '__main__':
	main()
