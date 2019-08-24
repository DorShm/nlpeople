import nltk
from nltk.stem import WordNetLemmatizer

def map_tags_to_ids(tags, include_unk):
  tag_to_id = {}
  i = 0

  for i, tag in enumerate(tags):
    tag_to_id[tag] = i

  if include_unk:
    tag_to_id['UNK'] = i + 1

  return tag_to_id

'''
  Document this!
'''
def get_context_pos(context, pos_tag_to_id):
  text = nltk.word_tokenize(context)
  pos_context = nltk.pos_tag(text)
  return [pos_tag_to_id.get(pos_tag, pos_tag_to_id['UNK']) for _, pos_tag in pos_context]

'''
  Document this!
'''
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

'''
  Document this!
'''
def extract_chunk_data(chunk, ner_tag_to_id):
  label = ner_tag_to_id[chunk.label()] if hasattr(chunk, 'label') else ner_tag_to_id['O']
  chunk_length = len(chunk) if isinstance(chunk, nltk.Tree) else 1

  return label, chunk_length

'''
  Document this!
'''
def get_match_vectors(context, question):
  match_vectors = []

  context_words = nltk.word_tokenize(context)
  question_words = nltk.word_tokenize(question)

  for context_word in context_words:
    match_vector = get_match_vector(context_word, question_words)
    match_vectors.append(match_vector)

  return match_vectors


'''
  Document this!
'''
def get_match_vector(matched_word, sentence):
  lower_sentence = list(map(lambda word: str.lower(word), sentence))

  lemmatizer = WordNetLemmatizer()
  matched_word_lemma = lemmatizer.lemmatize(str.lower(matched_word))
  sentence_lemmas = list(map(lambda word: lemmatizer.lemmatize(str.lower(word)), sentence))

  original_match = 1. if matched_word in sentence else 0.
  lower_match = 1. if str.lower(matched_word) in lower_sentence else 0.
  lemma_match = 1. if matched_word_lemma in sentence_lemmas else 0.

  return [original_match, lower_match, lemma_match]

'''
  Document this!
'''
def get_alignment_vector():
  pass

def get_text_embeddings(text, embeddings_module):
  return [embeddings_module[word] for word in text]