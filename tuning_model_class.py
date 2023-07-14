from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import mlflow, os, nltk, random, re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from nltk.corpus import words

nltk.download('words')
nltk.download('punkt')
vocabulary = words.words()

class w2v_tuning:
    def __init__(self, vectorsize=300, window=5, min_count=1, workers=4):
        self.model = None
        