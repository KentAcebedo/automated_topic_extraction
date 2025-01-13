#!/usr/bin/env python
# coding: utf-8

import nltk
from sklearn.datasets import fetch_20newsgroups
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
import spacy
import pandas as pd
from PyPDF2 import PdfReader
from gensim import corpora, models
from gensim.parsing.preprocessing import preprocess_string

# Ensure nltk downloads required resources
nltk.download("stopwords")
nltk.download('wordnet')

# Fetch the dataset
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)

print(list(newsgroups_train.target_names))

# Sample news
print(newsgroups_train.data[:2])
print(newsgroups_train.filenames.shape, newsgroups_train.target.shape)

# Define the stemmer and lemmatizer
stemmer = SnowballStemmer("english")

# Sample lemmatization and stemming
print(WordNetLemmatizer().lemmatize('went', pos='v'))

original_words = [
    'caresses', 'flies', 'dies', 'mules', 'denied', 'died', 'agreed', 'owned',
    'humbled', 'sized', 'meeting', 'stating', 'siezing', 'itemization', 'sensational',
    'traditional', 'reference', 'colonizer', 'plotted'
]
singles = [stemmer.stem(plural) for plural in original_words]

print(pd.DataFrame(data={'original word': original_words, 'stemmed': singles}))

# Define preprocessing functions
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# Example of preprocessing
doc_sample = 'This disk has failed many times. I would like to get it replaced.'
print("Original document: ", doc_sample.split(' '))
print("Tokenized and lemmatized document: ", preprocess(doc_sample))

# Process the training documents
processed_docs = [preprocess(doc) for doc in newsgroups_train.data]
print(processed_docs[:2])

# Create a dictionary
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

# Create a bag of words corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

document_num = 20
bow_doc_x = bow_corpus[document_num]
for i in range(len(bow_doc_x)):
    print(f"Word {bow_doc_x[i][0]} (\"{dictionary[bow_doc_x[i][0]]}\") appears {bow_doc_x[i][1]} time.")

# Train an LDA model
lda_model = gensim.models.LdaMulticore(
    bow_corpus,
    num_topics=8,
    id2word=dictionary,
    passes=10,
    workers=2
)

for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")

# Load and preprocess a PDF document
pdf_file_path = "C:\\Users\\acer\\Desktop\\NLPP\\final123.pdf"
with open(pdf_file_path, "rb") as pdf_file:
    pdf_reader = PdfReader(pdf_file)
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        pdf_text += page_text

unseen_document = pdf_text
print(unseen_document)

# Preprocess and create a dictionary and corpus for the PDF text
preprocessed_text = preprocess_string(pdf_text)
dictionary = corpora.Dictionary([preprocessed_text])
corpus = [dictionary.doc2bow(preprocessed_text)]

# Train an LDA model for the PDF text
lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=2)
for topic in lda_model.print_topics():
    print(topic)

# Analyze the unseen document
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print(f"Score: {score}\n Topic: {lda_model.print_topic(index, 10)}")

# Get the highest-scoring topic and its representative sentence
topic_distribution = lda_model[bow_vector]
highest_score_topic, highest_score = max(topic_distribution, key=lambda x: x[1])
nlp = spacy.load("en_core_web_sm")
top_words = lda_model.print_topic(highest_score_topic, topn=5)
top_words_list = [word.strip().split('*')[1].strip('\"') for word in top_words.split('+')]
lemmatized_words = [token.lemma_ for token in nlp(" ".join(top_words_list))]

representative_sentence = f"This document is about {' '.join(lemmatized_words)}."
print(f"Highest Score: {highest_score}\t{representative_sentence}")
