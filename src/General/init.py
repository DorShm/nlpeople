import nltk
import src.General.settings as settings
import gensim.downloader as api
from gensim.models import Word2Vec

words_embeddings = None

def init():
  global words_embeddings

  settings.init()
  if bool(int(settings.config['preprocessing']['download'])):
    download_nltk_datasets()
    words_embeddings = api.load(settings.config['wordsembeddings']['modle_name'])
    words_embeddings.save("../assets/word2vec.model")
  else:
    words_embeddings = Word2Vec.load("../assets/word2vec.model")

def download_nltk_datasets():
  nltk.download('tagsets')
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  nltk.download('maxent_ne_chunker')
  nltk.download('words')
  nltk.download('wordnet')