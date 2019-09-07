import ast
import json

from src.General import settings, init
from src.Preprocess import preprocess
from src.SQuADModel import SQuADModel


def main():
    logger = init.init()
    if ast.literal_eval(settings.config['preprocessing']['preprocess']):
        preprocess.preprocess(ast.literal_eval(settings.config['preprocessing']['save_preprocess_data']))

    with open(settings.config['general']['data_file'], 'r') as f:
      data = json.load(f)['data']
    squad_model = SQuADModel(init.words_embeddings, settings.config, logger)
    logger.info('Starting to train the network')
    for epoch in range(int(settings.config['general']['epoches'])):
        logger.info(f'Starting epoch {epoch}')
        for paragraph in data:
            for question in paragraph['qas']:
                squad_model.update(paragraph, question)


if __name__ == '__main__':
    main()
