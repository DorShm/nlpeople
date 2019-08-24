import nltk
import src.General.settings as settings
import gensim.downloader as api

words_embeddings = None

def init():
  global words_embeddings

  download_nltk_datasets()
  settings.init()
  words_embeddings = api.load(settings.config['wordsembeddings']['model_name'])

def download_nltk_datasets():
  nltk.download('tagsets')
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('maxent_ne_chunker')
  nltk.download('words')
  nltk.download('wordnet')