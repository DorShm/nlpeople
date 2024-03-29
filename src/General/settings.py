import nltk
from configparser import ConfigParser
from General.utils import *

#/content/drive/My Drive/NLP/Project
config_file_path = "../assets/config.ini"
config = None
pos_tags = None
ner_tags = None

def init():
  initialize_config()
  initialize_hardcoded()

def initialize_config():
  global config
  global config_file_path

  config = ConfigParser()
  config.read(config_file_path)
  config = config_to_dict(config)

def initialize_hardcoded():
  global pos_tags
  global ner_tags

  pos_tags = nltk.load('help/tagsets/upenn_tagset.pickle')
  ner_tags = ['GSP', 'LOCATION', 'GPE', 'ORGANIZATION', 'PERSON', 'O', 'PERSON', 'FACILITY']