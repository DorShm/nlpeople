import json
import torch
from torch import nn
from Encoders.GermanEnglishCoVe import GermanEnglishCoVe
from src.Encoders.LexiconEncoder import LexiconEncoder
from src.General.FeedForward import FeedForward


class QAModule(nn.Module):
  def __init__(self, words_embeddings, config):
    super(QAModule, self).__init__()
    self.lexicon_config = config['lexicon']
    self.contextual_config = config['contextual']
    self.config = config['qamodule']
    self.lexicon_encoder = LexiconEncoder(words_embeddings, self.lexicon_config)
    #TODO : change to one model for the German English model instead of two
    self.question_contextual_encoder = GermanEnglishCoVe(self.contextual_config)
    self.paragraph_contextual_encoder = GermanEnglishCoVe(self.contextual_config)
    self.data = None
    with open(self.config['data_file'], 'r') as f:
      self.data = json.load(f)['data']

  def forward(self, sentence, question):
    paragraph_vector, question_vector, paragraph_emb, question_emb = self.lexicon_encoder(sentence, question)

    # TODO : add german english model
    # TODO : create a ContextualEncoder Class - Use ContextualEncoder
    question_vector = self.question_contextual_encoder(question_vector)
    sentence_vector = self.paragraph_contextual_encoder(paragraph_vector)

    # TODO: create the memory layer

    # TODO: create the finale GRU layer
    return question_vector, sentence_vector
