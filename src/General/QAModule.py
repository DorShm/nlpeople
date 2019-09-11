import json
from torch import nn
from Layers.MemoryLayer import MemoryLayer
from Layers.AnswerLayer import AnswerLayer
from Encoders.GermanEnglishCoVe import GermanEnglishCoVe
from Encoders.LexiconEncoder import LexiconEncoder
from Encoders.ContextEncoder import ContextEncoder
from General.Networks import LinearSelfAttn


class QAModule(nn.Module):
  def __init__(self, words_embeddings, config, cuda_on):
    super(QAModule, self).__init__()
    # configurations
    self.lexicon_config = config['lexicon']
    self.german_english_cove_config = config['german_english_cove']
    self.contextual_config = config['contextual']
    self.memory_config = config['memory_layer']
    self.answer_config = config['answer_layer']
    self.cuda_on = cuda_on

    # networks
    self.lexicon_encoder: LexiconEncoder = LexiconEncoder(words_embeddings, self.lexicon_config, self.cuda_on)
    self.german_english_cove: GermanEnglishCoVe = GermanEnglishCoVe(self.german_english_cove_config)

    self.contextual_config['input_size'] = \
      self.german_english_cove.output_size + self.lexicon_encoder.output_size
    self.contextual_config['cove_size'] = self.german_english_cove.output_size

    self.paragraph_contextual_encoder: ContextEncoder = ContextEncoder(self.contextual_config)
    self.question_contextual_encoder: ContextEncoder = ContextEncoder(self.contextual_config)
    self.memory_layer: MemoryLayer = MemoryLayer(self.memory_config,
                                                 self.question_contextual_encoder.layer_2.hidden_size, self.cuda_on)
    self.linear_self_attention: LinearSelfAttn = LinearSelfAttn(self.question_contextual_encoder.output_size)
    self.answer_layer: AnswerLayer = AnswerLayer(self.answer_config, self.memory_layer.output_size,
                                    self.question_contextual_encoder.output_size)

    if self.cuda_on:
      self.set_cuda()

  def set_cuda(self):
    self.lexicon_encoder.cuda()
    self.german_english_cove.cuda()
    self.paragraph_contextual_encoder.cuda()
    self.question_contextual_encoder.cuda()
    self.memory_layer.cuda()
    self.linear_self_attention.cuda()
    self.answer_layer.cuda()

  def forward(self, sentence, question):
    # Lexicon Layer
    paragraph_vector, question_vector, paragraph_emb, question_emb = self.lexicon_encoder(sentence, question)

    # COVE Layer
    question_cove_vector_l1, question_cove_vector_l2 = self.german_english_cove(question_emb)
    paragraph_cove_vector_l1, paragraph_cove_vector_l2 = self.german_english_cove(paragraph_emb)

    # Contextual Layer
    question_vector = self.question_contextual_encoder(question_vector, question_cove_vector_l1,
                                                       question_cove_vector_l2)
    paragraph_vector = self.paragraph_contextual_encoder(paragraph_vector, paragraph_cove_vector_l1,
                                                        paragraph_cove_vector_l2)

    memory = self.memory_layer(question_vector, paragraph_vector)

    # TODO: create the finale GRU layer
    GRU_initial_state = self.linear_self_attention(question_vector)
    start, end = self.answer_layer(memory, GRU_initial_state)
    return start, end
