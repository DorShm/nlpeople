import json

import nltk
from src.General import utils
from src.General import settings

pos_tag_to_id = None
ner_tag_to_id = None

'''
  Document this!
'''
def preprocess(save_preproces_data=False):
  init_preprocess()

  instances = read_data()
  preprocessed_instances = []

  for instance in instances:
    for paragraph in instance["paragraphs"]:
      instance = preprocess_paragraph(paragraph)

      preprocessed_instances.append(instance)

  if save_preproces_data:
    save_preprocessed_data(preprocessed_instances)

  return preprocessed_instances

'''
  Document this!
'''
def init_preprocess():
  global pos_tag_to_id
  global ner_tag_to_id

  pos_tag_to_id = utils.map_tags_to_ids(settings.pos_tags, True)
  ner_tag_to_id = utils.map_tags_to_ids(settings.ner_tags, False)

def read_data():
  with open(settings.config['preprocessing']['data'], 'r') as j:
    instances = json.load(j)['data']

  return instances

def save_preprocessed_data(preprocessed_instances):
  with open(settings.config['preprocessing']["output_file"], 'w') as output:
    json.dump({'data': preprocessed_instances}, output)

'''
  Document this!
'''
def preprocess_paragraph(paragraph):
  instance = {
    "context": nltk.word_tokenize(paragraph["context"]),
    "context_pos": utils.get_context_pos(paragraph["context"], pos_tag_to_id),
    "context_ner": utils.get_context_ner(paragraph["context"], ner_tag_to_id),
    "qas": []
  }
  # instance['context_emb'] = get_text_embeddings(paragraph['context'], embeddings_model)

  for question in paragraph["qas"]:
    instance_question = preprocess_question(paragraph, question)

    instance["qas"].append(instance_question)

  return instance


'''
  Document this!
'''
def preprocess_question(paragraph, question):
  instance_question = {
    "question": nltk.word_tokenize(question["question"]),
    "exact_match": utils.get_match_vectors(paragraph["context"], question["question"])}

  answer_details = question["answers"][0]

  instance_question['answer_start'], instance_question['answer_end'] = \
    utils.find_answer_word_index(paragraph['context'], answer_details['text'], answer_details['answer_start'])

  return instance_question