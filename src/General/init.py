import nltk
import src.General.settings as settings
import gensim.downloader as api
import pickle

words_embeddings = None


def init():
    global words_embeddings

    settings.init()
    if bool(int(settings.config['preprocessing']['download'])):
        download_nltk_datasets()
        words_embeddings = api.load(settings.config['wordsembeddings']['modle_name'])
        pickle.dump(words_embeddings, open("../assets/word2vec.model", 'wb'))
    else:
        words_embeddings = pickle.load(open("../assets/word2vec.model", 'rb'))


def download_nltk_datasets():
    nltk.download('tagsets')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('wordnet')
