import ast
import math

import torch
import numpy as np
import logging
from torch import Tensor, optim as optimizer
from torch.autograd import Variable
from torch.nn import utils, functional as F

from General.utils import ModelLoss
from General.QAModule import QAModule


# noinspection PyArgumentList, PyTypeChecker
class SQuADModel:
  def __init__(self, words_embeddings, config):
    self.config = config['squad_model']
    self.cuda_on = torch.cuda.is_available() and ast.literal_eval(self.config['cuda_on'])

    self.qa_module: QAModule = QAModule(words_embeddings, config, self.cuda_on)
    if self.cuda_on:
      self.set_cuda()

    self.model_loss = ModelLoss()
    self.logger = logging.getLogger('nlpeople_logger')
    parameters = [parameter for parameter in self.qa_module.parameters() if parameter.requires_grad]

    self.optimizer = optimizer.Adamax(parameters, float(self.config['learning_rate']))

  def set_cuda(self):
    self.qa_module.cuda()

  def update(self, paragraph, question):
    self.qa_module.train()

    start_tensor = self.set_tensor_cuda(torch.tensor([question['answer_start']]))
    end_tensor: Tensor = self.set_tensor_cuda(torch.tensor([question['answer_end']]))

    start_label = Variable(start_tensor)
    end_label = Variable(end_tensor)

    start, end = self.qa_module(paragraph, question)

    loss = F.cross_entropy(start, start_label) + F.cross_entropy(end, end_label)

    self.model_loss.calculate_loss(loss)

    loss.backward()

    # Clip gradients to avoid gradients explosion
    utils.clip_grad_norm_(self.qa_module.parameters(), int(self.config['max_norm']))

    # Calculate parameters with optimizer step
    self.optimizer.step()

  def set_tensor_cuda(self, tensor) -> Tensor:
    """

    :rtype: Tensor
    """
    return tensor.cuda() if self.cuda_on else tensor

  def predict(self, paragraph, question):
    self.qa_module.eval()

    start_prediction, end_prediction = self.qa_module(paragraph, question)

    start = np.argmax(start_prediction.detach().cpu().numpy())
    end = np.argmax(end_prediction.detach().cpu().numpy())

    return start, end

  def eval(self, predictions, labels) -> float:
    predictions = np.array(predictions)
    labels = np.array(labels)

    exact_match = (predictions == labels).all(axis=1).mean()
    half_match = (predictions == labels).any(axis=1).mean()

    return exact_match, half_match

  def load(self, epoch='best'):
    """
    Load trained model state dict from file
    :param epoch: epoch number
    """
    self.logger.info(f'Using model loaded from {self.config["squad_model_path"].format(epoch)}')

    state_dict = torch.load(self.config["squad_model_path"].format(epoch))
    self.qa_module.load_state_dict(state_dict)

  def save(self, epoch='best'):
    """
    Save trained model state dict
    :param epoch: epoch number
    """
    torch.save(self.qa_module.state_dict(), self.config['squad_model_path'].format(epoch))

    self.logger.info(f'Saved SQuAD model in {self.config["squad_model_path"].format(epoch)}')