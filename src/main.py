import ast
import json
import logging

from tqdm import tqdm

from General import settings, init
from Preprocess import preprocess
from SQuADModel import SQuADModel


def main():
    init.init()
    logger = logging.getLogger('nlpeople_logger')
    is_preprocessing = ast.literal_eval(settings.config['preprocessing']['preprocess'])
    if is_preprocessing:
        preprocess.preprocess(ast.literal_eval(settings.config['preprocessing']['save_preprocess_data']))

    train, test = load_preprocessed_data(logger, float(settings.config['general']['split_ratio']))

    squad_model = SQuADModel(init.words_embeddings, settings.config)
    logger.info(f'Starting to train the network using data from {settings.config["preprocessing"]["data"]} doing {settings.config["general"]["epoches"]} epochs')
    for epoch in range(int(settings.config['general']['epochs'])):
        logger.info(f'Starting epoch {epoch}')
        for paragraph in tqdm(train, total=len(train)):
            for question in paragraph['qas']:
                squad_model.update(paragraph, question)

        squad_model.save(epoch)


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
