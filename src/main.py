import ast
import json
import logging

from tqdm import tqdm

from General import settings, init
from Preprocess import preprocess
from SQuADModel import SQuADModel
from test_model import run_model_test


def main():
    init.init()
    logger = logging.getLogger('nlpeople_logger')
    is_preprocessing = ast.literal_eval(settings.config['preprocessing']['preprocess'])
    if is_preprocessing:
        preprocess.preprocess(ast.literal_eval(settings.config['preprocessing']['save_preprocess_data']))

    train, test = load_preprocessed_data(logger, float(settings.config['general']['split_ratio']))

    squad_model = SQuADModel(init.words_embeddings, settings.config)
    best_accuracy = 0.

    # TODO: Remove before submission
    test_model = ast.literal_eval(settings.config['general']['test_model'])

    if test_model:
        squad_model.load(settings.config['general']['loaded_epoch'])

        run_model_test(squad_model, train)
    else:
        logger.info(
            f'Starting to train the network using data from {settings.config["preprocessing"]["data"]} doing {settings.config["general"]["epochs"]} epochs')
        for epoch in range(1, int(settings.config['general']['epochs']) + 1):
            logger.info(f'Starting epoch {epoch} out of {settings.config["general"]["epochs"]} epochs')
            for paragraph in tqdm(train, total=len(train)):
                for question in paragraph['qas']:
                    squad_model.update(paragraph, question)

            squad_model.save(epoch)

            labels = []
            predictions = []

            logger.info(f'Starting evaluation on epoch {epoch}')
            # Predict
            for paragraph in tqdm(test, total=len(test)):
                for question in paragraph['qas']:
                    labels.append([question['answer_start'], question['answer_end']])

                    start, end = squad_model.predict(paragraph, question)
                    predictions.append([start, end])

            # Evaluate model accuracy
            em_accuracy, hm_accuracy = squad_model.eval(predictions, labels)

            logger.info(f'Accuracy for epoch {epoch}: em - {em_accuracy}, hm - {hm_accuracy}')

            if em_accuracy > best_accuracy:
                logger.info(f'Epoch {epoch} has better accuracy')
                best_accuracy = em_accuracy
                squad_model.save()

def load_preprocessed_data(logger, split_ratio):
    logger.info(f"Loading preprocessed data from {settings.config['general']['data_file']}")
    with open(settings.config['general']['data_file'], 'r') as f:
        data = json.load(f)['data']
    split_index = int(len(data) * split_ratio)
    train = data[:split_index]
    test = data[split_index:]
    return train, test


if __name__ == '__main__':
    main()
