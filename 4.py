from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, regexp_tokenize
from nltk.stem import PorterStemmer
from string import punctuation

import numpy as np
import math
import os

porter = PorterStemmer()
corpus = ''

for root, _, files in os.walk('data/imdb'):
	for name in files:
		file_path = os.path.join(root, name)
		# print('File:', file_path)

		with open(file_path, 'r') as f:
			file_data = f.read()
			corpus += file_data

# Cleaning the corpus
print('Cleaning the corpus...')
clean_corpus = []
stop_punc = set(stopwords.words('english')).union(set(punctuation))
corpus = sent_tokenize(corpus)

for doc in corpus[:5]:
	# words = wordpunct_tokenize(doc)
	words = regexp_tokenize(doc, "[\w']+")
	words_lower = [w.lower() for w in words]
	print('words_lower:', words_lower)
	words_filtered = [w for w in words_lower if w not in stop_punc]
	print('words_filtered:', words_filtered)
	words_stemmed = [porter.stem(w) for w in words_filtered]
	print('words_stemmed:', words_stemmed)
	clean_corpus.append(words_stemmed)
	print()

# Create vocabulary
print('Creating the vocabulary...')
vocab_set = set()

for doc in clean_corpus:
	for word in doc:
		vocab_set.add(word)

vocab = list(vocab_set)
vocabulary = list(zip(vocab, range(len(vocab))))

print('Vocabulary:', vocabulary)
print('Feature vector size: ', len(vocab))


# with open('out.txt', 'w') as f:
# 	for w in vocabulary:
# 		f.write(':'.join(map(str, w)) + '\n')


def occ_score(word, doc):
	return 1 if word in doc else 0


def numocc_score(word, doc):
	return doc.count(word)


def freq_score(word, doc):
	return doc.count(word) / len(doc) if len(doc) > 0 else 0


def get_bow():
	# 1: Bag of Words model sa 3 različita scoringa
	np.set_printoptions(precision=2, linewidth=200)
	print('Creating BOW features...')

	for score_fn in [occ_score, numocc_score, freq_score]:
		X = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)

		for doc_idx in range(len(clean_corpus)):
			doc = clean_corpus[doc_idx]

			for word_idx in range(len(vocab)):
				word = vocab[word_idx]
				cnt = score_fn(word, doc)
				X[doc_idx][word_idx] = cnt

		print('X:')
		print(X)
		print()


def get_bigrams():
	# 2. Bigram model
	for doc in clean_corpus:
		bigrams = []

		for i in range(len(doc) - 1):
			bigram = doc[i] + ' ' + doc[i + 1]
			bigrams.append(bigram)

		print('Bigrams:', bigrams)


def get_tf_idf():
	# 3. TF-IDF
	# Racunamo IDF mapu
	print('Calculating the IDF table...')
	doc_counts = dict()

	for word in vocab:
		doc_counts[word] = 0

		for doc in clean_corpus:
			if word in doc:
				doc_counts[word] += 1

	print('Doc counts:')
	print(doc_counts)
	idf_table = dict()

	for word in vocab:
		idf = math.log10(len(corpus) / doc_counts[word])
		idf_table[word] = idf

	print('IDF table:')
	print(idf_table)

	# Isti kod kao kod BOW
	def tfidf_score(word, doc):
		tf = freq_score(word, doc)
		idf = idf_table[word]
		return tf * idf

	print('Creating TF-IDF features...')
	X = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)

	for doc_idx in range(len(clean_corpus)):
		doc = clean_corpus[doc_idx]
		for word_idx in range(len(vocab)):
			word = vocab[word_idx]
			cnt = tfidf_score(word, doc)
			X[doc_idx][word_idx] = cnt

	print('X:')
	print(X)


# TODO: MODELOVANJE I SIMULACIJA
get_bow()
get_bigrams()
get_tf_idf()
