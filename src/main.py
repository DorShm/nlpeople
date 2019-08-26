from src.General import settings, init
from src.General.QAModule import QAModule
from src.Preprocess import preprocess


def main():
    init.init()
    # TODO : use preprocessing only when required
    #preprocess.preprocess(True)
    qa_module = QAModule(init.words_embeddings, settings.config)

    for sentence in qa_module.data:
        for question in sentence['qas']:
            x, y = qa_module(sentence, question)


if __name__ == '__main__':
    main()
