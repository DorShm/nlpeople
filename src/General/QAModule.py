import json
import torch
from torch import nn
from src.Encoders.ContextEncoder import ContextEncoder
from src.Encoders.LexiconEncoder import LexiconEncoder
from src.General.FeedForward import FeedForward


class QAModule(nn.Module):
  def __init__(self, words_embeddings, config):
    super(QAModule, self).__init__()
    self.lexicon_config = config['lexicon']
    self.contextual_config = config['contextual']
    self.config = config['qamodule']

    self.qFFN = FeedForward(int(self.config['question_FFN_input_size']),
                            int(self.config['question_FFN_hidden_size']),
                            int(self.config['question_FFN_second_hidden_size']))
    self.dFFN = FeedForward(int(self.config['paragraph_FFN_input_size']),
                            int(self.config['paragraph_FFN_hidden_size']),
                            int(self.config['paragraph_FFN_second_hidden_size']))
    self.lexicon_encoder = LexiconEncoder(words_embeddings, self.lexicon_config)
    self.question_contextual_encoder = ContextEncoder(self.contextual_config)
    self.paragraph_contextual_encoder = ContextEncoder(self.contextual_config)
    self.data = None
    with open(self.config['data_file'], 'r') as f:
      self.data = json.load(f)['data']

  def forward(self, sentence, question):
    sentence_emb = self.lexicon_encoder.get_sentence_embeddings(sentence['context'])
    sentence_pos = sentence['context_pos']
    sentence_ner = sentence['context_ner']
    sentence_match = torch.stack([torch.tensor(match) for match in question['exact_match']])
    question_emb = self.lexicon_encoder.get_sentence_embeddings(question['question'])
    sentence_vector = self.lexicon_encoder.create_doc_vector(sentence_emb, sentence_pos,
                                                             sentence_ner, sentence_match,
                                                             question_emb, sentence['context'],
                                                             question['question'])

    question_vector = self.qFFN(question_emb)
    sentence_vector = self.dFFN(sentence_vector)

    question_vector = self.question_contextual_encoder(question_vector)
    sentence_vector = self.paragraph_contextual_encoder(sentence_vector)

    return question_vector, sentence_vector
