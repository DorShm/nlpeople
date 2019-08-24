import torch
import torch.nn.functional as functional


class LexiconEncoder:
  def __init__(self, words_embeddings, config):
    self.words_embeddings = words_embeddings
    self.pos_embeddings = torch.nn.Embedding(int(config['pos_embeddings_size']),
                                             int(config['pos_embeddings_output_size']))
    self.ner_embeddings = torch.nn.Embedding(int(config['ner_embeddings_size']),
                                             int(config['ner_embeddings_output_size']))
    self.g = torch.nn.Linear(int(config['similarity_linear_nn_input_size']),
                             int(config['similarity_linear_nn_output_size']))

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
    return torch.tensor(sum(similarity))

  '''
  return a tensor with the embeddings of a sentence
  '''

  def get_sentence_embeddings(self, sentence):
    return torch.stack([self.get_word_embeddings(word) for word in sentence])

  '''
  return a tensor that represent the given word
  '''

  def get_word_embeddings(self, word):
    return torch.tensor(
      self.words_embeddings[word.lower()] if word.lower() in self.words_embeddings.vocab else torch.tensor(
        self.words_embeddings['unk']))

  '''
  return a matrix where each row 'j' is a tensor that represent the pos of the word p_j in the document 
  '''

  def get_pos_matrix(self, pos_list):
    pos_emb = [self.pos_embeddings(torch.tensor(word_pos)) for word_pos in pos_list]
    return torch.stack(pos_emb)

  '''
  return a matrix where each row 'j' is a tensor that represent the ner of the word p_j in the document 
  '''

  def get_ner_matrix(self, ner_list):
    ner_emb = [self.ner_embeddings(torch.tensor(word_ner)) for word_ner in ner_list]
    return torch.stack(ner_emb)

  def create_doc_vector(self, sentence_embeddings, context_pos, context_ner, context_match, question_emb, sentence,
                        question):
    pos_emb = self.get_pos_matrix(context_pos)
    ner_emb = self.get_ner_matrix(context_ner)
    match_emb = torch.stack([torch.tensor(match) for match in context_match])
    similarity_matrix = self.get_words_similarity_value(sentence, question)
    embeddings_vector = torch.cat((sentence_embeddings, pos_emb, ner_emb, match_emb, similarity_matrix), 1)
    return embeddings_vector