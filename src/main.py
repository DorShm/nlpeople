import ast
import json
import logging

from tqdm import tqdm

from src.General import settings, init
from src.Preprocess import preprocess
from src.SQuADModel import SQuADModel


def main():
    init.init()
    logger = logging.getLogger('nlpeople_logger')
    preprocess = ast.literal_eval(settings.config['preprocessing']['preprocess'])
    if preprocess:
        preprocess.preprocess(ast.literal_eval(settings.config['preprocessing']['save_preprocess_data']))

    train, test = load_preprocessed_data(logger, float(settings.config['general']['split_ratio']))

    squad_model = SQuADModel(init.words_embeddings, settings.config)
    logger.info('Starting to train the network')
    for epoch in range(int(settings.config['general']['epoches'])):
        logger.info(f'Starting epoch {epoch}')
        for paragraph in tqdm(train, total=len(train)):
            for question in paragraph['qas']:
                squad_model.update(paragraph, question)

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
