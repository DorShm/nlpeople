import json
import torch
from torch import nn

from Layers.MemoryLayer import MemoryLayer
from src.Encoders.GermanEnglishCoVe import GermanEnglishCoVe
from src.Encoders.LexiconEncoder import LexiconEncoder
from src.Encoders.ContextEncoder import ContextEncoder


class QAModule(nn.Module):
  def __init__(self, words_embeddings, config):
    super(QAModule, self).__init__()
    # configurations
    self.lexicon_config = config['lexicon']
    self.german_english_cove_config = config['german_english_cove']
    self.contextual_config = config['contextual']
    self.config = config['qamodule']

    # networks
    self.lexicon_encoder = LexiconEncoder(words_embeddings, self.lexicon_config)
    self.german_english_cove = GermanEnglishCoVe(self.german_english_cove_config)
    self.paragraph_contextual_encoder = ContextEncoder(self.contextual_config)
    self.question_contextual_encoder = ContextEncoder(self.contextual_config)

    self.data = None
    with open(self.config['data_file'], 'r') as f:
      self.data = json.load(f)['data']

  def forward(self, sentence, question):
    # Lexicon Layer
    paragraph_vector, question_vector, paragraph_emb, question_emb = self.lexicon_encoder(sentence, question)

    # COVE Layer
    question_cove_vector_l1, question_cove_vector_l2 = self.german_english_cove(question_emb)
    paragraph_cove_vector_l1, paragraph_cove_vector_l2 = self.german_english_cove(paragraph_emb)

    # Contextual Layer
    question_vector = self.question_contextual_encoder(question_vector, question_cove_vector_l1,
                                                       question_cove_vector_l2)
    sentence_vector = self.paragraph_contextual_encoder(paragraph_vector, paragraph_cove_vector_l1,
                                                        paragraph_cove_vector_l2)

    # TODO: create the memory layer
    memory_layer = MemoryLayer(self.memory_config)

    # TODO: create the finale GRU layer
    return question_vector, sentence_vector
