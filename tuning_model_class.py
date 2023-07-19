import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, CoherenceModel
from gensim.models.keyedvectors import KeyedVectors
import mlflow, os, nltk, random, re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import nltk
from nltk.corpus import wordnet, stopwords, words
from nltk.stem import WordNetLemmatizer

nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = stopwords.words('english')
vocabulary = words.words()

class w2v_tuning:
    def __init__(self, vectorsize=300, window=5, min_count=1, workers=4):
        self.model = None
        self.pre_train_file_path  = None
        self.w2v_file_path = None
        self.lemmatizer = WordNetLemmatizer()
        self.corpus = []
        self.vectorsize = vectorsize
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def preprocess_text(self,sentences):

        lemmatized_sentences = []
        bigram = gensim.models.Phrases(sentences, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        
        data_words_nostops = [[word for word in doc if word not in stop_words] for doc in sentences]
        data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]

        for sentence in data_words_bigrams:
            lemmatized_tokens = [self.lemmatizer.lemmatize(token, wordnet.VERB) for token in sentence]
            lemmatized_sentences.append(lemmatized_tokens)
        return lemmatized_sentences
    
    def import_corpus(self,file_path, delimiter='\t'):
        result = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace and newlines
                sublist = line.split(delimiter)
                result.append(sublist)
        self.corpus += self.preprocess_text(result)
        return result
    
    def test_Accuracy(self, test_file_path):
        with open(test_file_path, 'r') as file:
            lines = file.readlines()
        total_questions = 0
        correct_predictions = 0
        random_amount = 1000
        random.seed(42)
        for line in random.sample(lines, random_amount):
            if line.startswith(':'):
                continue  # Skip comment lines and category headers
            question, expected_answer = [word.lower() for word in line.split()[:2]]
            choices = [word.lower() for word in line.split()[2:]]
            try:
                predicted_answer = self.model.wv.most_similar(positive=[question, choices[1]], negative=[expected_answer])[0][0]
                if predicted_answer == choices[0]:
                    correct_predictions += 1
            except KeyError:
                continue  # Skip questions with out-of-vocabulary words

            total_questions += 1

        accuracy = correct_predictions / total_questions * 100
        return accuracy
    
    def set_vector_path(self,w2v_file,pre_train_file,folder_path='src'):

        self.pre_train_file_path = os.path.join(folder_path, pre_train_file)
        self.w2v_file_path = os.path.join(folder_path, w2v_file)
    
    def CreateW2V_Model(self):

        RUN_NAME = "Create Word2Vec Model"

        if mlflow.active_run():
            mlflow.end_run()
        
        with mlflow.start_run(run_name=RUN_NAME):
            # Check exist model
            if os.path.isfile(self.pre_train_file_path):
                # Load word vectors
                print('load pre-train model')
                GloVe_vectors = KeyedVectors.load(self.pre_train_file_path)
            else:
                print('convert GloVe to Word2Vec')
                GloVe_vectors = KeyedVectors.load_word2vec_format(self.w2v_file_path, binary=False)

            # Initialize a Gensim Word2Vec model
            self.model = Word2Vec(vector_size=self.vectorsize, window=self.window, min_count=self.min_count, workers=self.workers)
            # Set the vocabulary and vectors of the model
            self.model.build_vocab_from_freq(GloVe_vectors.key_to_index)
            self.model.wv.key_to_index = GloVe_vectors.key_to_index
            self.model.wv.vectors = GloVe_vectors.vectors
            self.model.wv.index_to_key = GloVe_vectors.index_to_key
            
            if not os.path.isfile(self.pre_train_file_path):
                print("save pre-train word2vec model")
                self.model.wv.save(self.pre_train_file_path) 

            # Log parameters
            mlflow.log_param("vectorsize", self.vectorsize)
            mlflow.log_param("window", self.window)
            mlflow.log_param("min_count", self.min_count)
            mlflow.log_param("workers", self.workers)

            # Log model artifact
            mlflow.log_artifact(self.pre_train_file_path)

    def train_with_corpus(self, RUN_NAME, new_corpus, epochs):
    
        if mlflow.active_run():
            mlflow.end_run()
            
        with mlflow.start_run(run_name=RUN_NAME):
            # Update the vocabulary with new words
            self.model.build_vocab(new_corpus, update=True)
            self.model.train(new_corpus, total_examples=len(new_corpus), epochs=epochs)

            # Log parameters
            mlflow.log_param("epochs", epochs)
            # Log model artifact
            # mlflow.sklearn.log_model(model, "word2vec_model")

            accuracy = self.test_Accuracy('src/questions-words.txt' )
            mlflow.log_metric("accuracy", accuracy)

    def save_model(self,file_path):

        self.model.wv.save(file_path)
        print('save model successful')

    def corpus_topic_score(self,num_topics=10):

        data_lemmatized = self.preprocess_text(self.corpus)
        id2word = corpora.Dictionary(data_lemmatized)

        LDA_corpus = [id2word.doc2bow(text) for text in data_lemmatized]
        lda_model = gensim.models.ldamodel.LdaModel(corpus=LDA_corpus,
                                            id2word=id2word,
                                            num_topics=num_topics, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
        
        print('\nPerplexity: ', lda_model.log_perplexity(LDA_corpus))  
        # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('Coherence Score: ', coherence_lda)


print('run')