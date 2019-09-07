import nltk
import logging
from nltk.stem import WordNetLemmatizer
from datetime import datetime

def map_tags_to_ids(tags, include_unk):
  tag_to_id = {}
  i = 0

  for i, tag in enumerate(tags):
    tag_to_id[tag] = i

  if include_unk:
    tag_to_id['UNK'] = i + 1

  return tag_to_id

def get_context_pos(context, pos_tag_to_id):
  text = nltk.word_tokenize(context)
  pos_context = nltk.pos_tag(text)
  return [pos_tag_to_id.get(pos_tag, pos_tag_to_id['UNK']) for _, pos_tag in pos_context]

def get_context_ner(context, ner_tag_to_id):
  context_ner = []
  words = nltk.word_tokenize(context)
  pos_tags = nltk.pos_tag(words)

  # Get all named entities based on POS tags
  named_entity_chunks = nltk.ne_chunk(pos_tags)

  for chunk in named_entity_chunks:
    label, chunk_length = extract_chunk_data(chunk, ner_tag_to_id)

    for i in range(chunk_length):
      context_ner.append(label)

  return context_ner

def extract_chunk_data(chunk, ner_tag_to_id):
  label = ner_tag_to_id[chunk.label()] if hasattr(chunk, 'label') else ner_tag_to_id['O']
  chunk_length = len(chunk) if isinstance(chunk, nltk.Tree) else 1

  return label, chunk_length

def get_match_vectors(context, question):
  match_vectors = []

  context_words = nltk.word_tokenize(context)
  question_words = nltk.word_tokenize(question)

  for context_word in context_words:
    match_vector = get_match_vector(context_word, question_words)
    match_vectors.append(match_vector)

  return match_vectors

def get_match_vector(matched_word, sentence):
  lower_sentence = list(map(lambda word: str.lower(word), sentence))

  lemmatizer = WordNetLemmatizer()
  matched_word_lemma = lemmatizer.lemmatize(str.lower(matched_word))
  sentence_lemmas = list(map(lambda word: lemmatizer.lemmatize(str.lower(word)), sentence))

  original_match = 1. if matched_word in sentence else 0.
  lower_match = 1. if str.lower(matched_word) in lower_sentence else 0.
  lemma_match = 1. if matched_word_lemma in sentence_lemmas else 0.

  return [original_match, lower_match, lemma_match]

def get_alignment_vector():
  pass

def get_text_embeddings(text, embeddings_module):
  return [embeddings_module[word] for word in text]

def find_answer_word_index(paragraph, answer, answer_start):
  """
  Finds the :answer word index in given :paragraph
  :param paragraph: which contains the :answer
  :param answer: answer text
  :param answer_start: answer start
  :return: (answer start word index, answer end word index)
  """
  answer_start_word = 0

  answer_tokens = nltk.word_tokenize(answer)
  paragraph_tokens = nltk.word_tokenize(paragraph)

  for word in paragraph_tokens:
    if answer_start <= 0:
      break

    answer_start -= len(word) + 1
    answer_start_word += 1

  return answer_start_word, answer_start_word + len(answer_tokens) - 1

def config_to_dict(config):
  """
  Converts a ConfigParser object into a dictionary.

  The resulting dictionary has sections as keys which point to a dict of the
  sections options as key => value pairs.
  """
  the_dict = {}
  for section in config.sections():
    the_dict[section] = {}
    for key, val in config.items(section):
      the_dict[section][key] = val
  return the_dict

class ModelLoss:
  def __init__(self):
    self.reset_counter()

  @property
  def average_loss(self):
    return self.loss_sum / self.loss_item_count if self.loss_item_count != 0 else 0

  def reset_counter(self):
    self.current_loss = 0
    self.loss_sum = 0
    self.loss_item_count = 0

  def calculate_loss(self, loss):
    self.current_loss = loss
    self.loss_sum += loss
    self.loss_item_count += 1

def init_logger():
  now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
  logger = logging.getLogger('nlpeople_logger')
  logger.setLevel(logging.DEBUG)

  fh = logging.FileHandler(f'..\logger\SQUAD_{now}.log')
  fh.setLevel(logging.DEBUG)

  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)

  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  fh.setFormatter(formatter)

  logger.addHandler(ch)
  logger.addHandler(fh)

  return logger
