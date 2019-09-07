import torch
import torch.nn.functional as functional
from torch import nn as nn

from src.General import utils
from src.General.Networks import FeedForward


class LexiconEncoder(nn.Module):
  def __init__(self, words_embeddings, config, cuda_on):
    super(LexiconEncoder, self).__init__()
    self.words_embeddings = words_embeddings
    self.cuda_on = cuda_on
    self.pos_embeddings = torch.nn.Embedding(int(config['pos_embeddings_size']),
                                             int(config['pos_embeddings_output_size']))

    self.ner_embeddings = torch.nn.Embedding(int(config['ner_embeddings_size']),
                                             int(config['ner_embeddings_output_size']))

    self.g = torch.nn.Linear(int(config['similarity_linear_nn_input_size']),
                             int(config['similarity_linear_nn_output_size']))

    self.qFFN = FeedForward(int(config['question_ffn_input_size']),
                            int(config['question_ffn_hidden_size']),
                            int(config['question_ffn_second_hidden_size']))

    self.dFFN = FeedForward(int(config['paragraph_ffn_input_size']),
                            int(config['paragraph_ffn_hidden_size']),
                            int(config['paragraph_ffn_second_hidden_size']))
    self.output_size = int(config['paragraph_ffn_second_hidden_size'])

  '''
  For each word create a 280 dimensional vector that represent the similarity between
  each word in the document to the whole words in the question
  '''

  def get_words_similarity_value(self, sentence, question):
    similarity_list = []
    for word in sentence:
      word_similarity = self.compute_similarity(word, question)
      word_vector = self.get_word_embeddings(word)
      word_vector = functional.relu(self.g(word_vector)) * word_similarity
      similarity_list.append(word_vector)
    return torch.stack(similarity_list)

  '''
  Compute the similarity between two given words using gensim
  '''

  def compute_similarity(self, word, question):
    similarity = []
    word_value = word.lower() if word.lower() in self.words_embeddings.vocab else 'unk'
    for question_word in question:
      question_value = question_word.lower() if question_word.lower() in self.words_embeddings.vocab else 'unk'
      similarity.append(self.words_embeddings.similarity(word_value, question_value))
    return utils.set_cuda(torch.tensor(sum(similarity)), self.cuda_on)

  '''
  return a tensor with the embeddings of a sentence
  '''

  def get_sentence_embeddings(self, sentence):
    return torch.stack([self.get_word_embeddings(word) for word in sentence])

  '''
  return a tensor that represent the given word
  '''

  def get_word_embeddings(self, word):
    return utils.set_cuda(torch.tensor(self.words_embeddings[word.lower()]
                          if word.lower() in self.words_embeddings.vocab
                          else self.words_embeddings['unk']), self.cuda_on)
  '''
  return a matrix where each row 'j' is a tensor that represent the pos of the word p_j in the document 
  '''

  def get_pos_matrix(self, pos_list):
    pos_emb = [self.pos_embeddings(utils.set_cuda(torch.tensor(word_pos), self.cuda_on)) for word_pos in pos_list]
    return torch.stack(pos_emb)

  '''
  return a matrix where each row 'j' is a tensor that represent the ner of the word p_j in the document 
  '''

  def get_ner_matrix(self, ner_list):
    ner_emb = [self.ner_embeddings(utils.set_cuda(torch.tensor(word_ner), self.cuda_on)) for word_ner in ner_list]
    return torch.stack(ner_emb)

  def create_doc_vector(self, sentence_embeddings, context_pos, context_ner, context_match, sentence,
                        question):
    pos_emb = self.get_pos_matrix(context_pos)
    ner_emb = self.get_ner_matrix(context_ner)
    similarity_matrix = self.get_words_similarity_value(sentence, question)
    embeddings_vector = torch.cat((sentence_embeddings, pos_emb, ner_emb, context_match, similarity_matrix), 1)
    return embeddings_vector

  def forward(self, paragraph, question):
    paragraph_emb = self.get_sentence_embeddings(paragraph['context'])
    paragraph_pos = paragraph['context_pos']
    paragraph_ner = paragraph['context_ner']
    paragraph_match = torch.stack([utils.set_cuda(torch.tensor(match), self.cuda_on) for match in question['exact_match']])
    question_emb = self.get_sentence_embeddings(question['question'])
    paragraph_vector = self.create_doc_vector(paragraph_emb, paragraph_pos,
                                                             paragraph_ner, paragraph_match,
                                                             paragraph['context'],
                                                             question['question'])
    size = question_emb.size()
    question_vector = self.qFFN(question_emb.view(1, size[0], size[1]))
    size = paragraph_vector.size()
    paragraph_vector = self.dFFN(paragraph_vector.view(1, size[0], size[1]))

    return paragraph_vector, question_vector, paragraph_emb, question_emb
