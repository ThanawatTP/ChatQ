import os, nltk
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

class word_embedding:
    def __init__(self, w2v_file_path = "../src/word2vec_files/w2v.42B.300d.txt", 
                 pre_train_file_path = "../src/word2vec_files/GloVe.wordvectors", 
                 vectorsize=300, window=5, min_count=1, workers=4):

        nltk.download('stopwords')
        nltk.download('words')
        nltk.download('wordnet')

        self.model = None
        self.pre_train_file_path  = pre_train_file_path
        self.w2v_file_path = w2v_file_path
        self.lemmatizer = WordNetLemmatizer()
        self.corpus = []
        self.vectorsize = vectorsize
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.stop_words = stopwords.words('english')

    def preprocess_text(self,sentences):
        
        lemmatized_sentences = []
        data_words_nostops = [[word for word in doc if word not in self.stop_words] for doc in sentences]

        # Cleaning word with lemmatized
        for sentence in data_words_nostops:
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
        # Add document to model's corpus
        self.corpus += self.preprocess_text(result)
        return result
        
    def create_model(self):

        # Check exist model
        if os.path.isfile(self.pre_train_file_path):
            # Load exist word vectors model
            print('loading pre-train model')
            GloVe_vectors = KeyedVectors.load(self.pre_train_file_path)
        else:
            print('converting GloVe to Word2Vec')
            GloVe_vectors = KeyedVectors.load_word2vec_format(self.w2v_file_path, binary=False)

        # Initialize a Gensim Word2Vec model
        self.model = Word2Vec(vector_size=self.vectorsize, window=self.window, min_count=self.min_count, workers=self.workers)
        # Set the vocabulary and vectors of the model
        self.model.build_vocab_from_freq(GloVe_vectors.key_to_index)
        self.model.wv.key_to_index = GloVe_vectors.key_to_index
        self.model.wv.vectors = GloVe_vectors.vectors
        self.model.wv.index_to_key = GloVe_vectors.index_to_key
        
        if not os.path.isfile(self.pre_train_file_path):
            print("saving pre-train word2vec model")
            self.model.wv.save(self.pre_train_file_path) 

    def train_with_corpus(self, new_corpus, epochs=1):
        # Update the vocabulary with new words
        print("Updating vocabs")
        self.model.build_vocab(new_corpus, update=True)
        self.model.train(new_corpus, total_examples=len(new_corpus), epochs=epochs)

    def save_model(self,file_path):
        self.model.wv.save(file_path)
        print('save model successful')
