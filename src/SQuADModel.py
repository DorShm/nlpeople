import ast
import torch

import torch
from torch import Tensor, optim as optimizer
from torch.autograd import Variable
from torch.nn import utils, functional as F

from src.General.utils import ModelLoss, LOGGER
from src.General.QAModule import QAModule


# noinspection PyArgumentList, PyTypeChecker
class SQuADModel:
  def __init__(self, words_embeddings, config):
    self.config = config['squad_model']
    self.qa_module: QAModule = QAModule(words_embeddings, config)
    self.cuda_on = ast.literal_eval(self.config['cuda_on'])
    self.model_loss = ModelLoss()
    parameters = [parameter for parameter in self.qa_module.parameters() if parameter.requires_grad]

    # TODO: Consider adding optimizer lr_scheduler
    self.optimizer = optimizer.Adamax(parameters, float(self.config['learning_rate']))


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

  def predict(self):
    self.qa_module.eval()
    pass

  # TODO: Check necessity
  def eval(self):
    pass

  def save(self):
    model_state_dict = dict([(key, value for key, value in self.qa_module.state_dict().items())])

    torch.save(model_state_dict, self.config['squad_model_path'])

    LOGGER.info(f'Saved SQuAD model in {self.config["squad_model_path"]}')