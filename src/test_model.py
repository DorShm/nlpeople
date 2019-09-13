from tqdm import tqdm

def run_model_test(squad_model, test):
  labels = []
  predictions = []

  for paragraph in tqdm(test, total=len(test)):
    for question in paragraph['qas']:
      labels.append([question['answer_start'], question['answer_end']])
      start, end = squad_model.predict(paragraph, question)
      predictions.append([start, end])

  em_accuracy, hm_accuracy = squad_model.eval(predictions, labels)
  print(f'em_accuracy: {em_accuracy}, hf_accuracy: {hm_accuracy}')