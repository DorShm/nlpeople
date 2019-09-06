import torch
from torch import Tensor, optim as optimizer
from torch.autograd import Variable
from torch.nn import utils, functional as F

from src.General.utils import ModelLoss
from src.General.QAModule import QAModule


# noinspection PyArgumentList
class SQuADModel:
  def __init__(self, words_embeddings, config):
    self.qa_module: QAModule = QAModule(words_embeddings, config)
    model_config = config['squad_model']
    self.cuda_on = bool(model_config['cuda_on'])
    self.model_loss = ModelLoss()

    # TODO: Add optimizer parameters to config learning_rate=0.002
    learning_rate = 0.002
    parameters = [parameter for parameter in self.qa_module.parameters() if parameter.requires_grad]
    # TODO: Consider adding optimizer lr_scheduler
    self.optimizer = optimizer.Adamax(parameters, learning_rate)


  def update(self, instance):
    self.qa_module.train()

    start_tensor = self.get_tensor(instance['start'])
    end_tensor: Tensor = self.get_tensor(instance['end'])

    start_label = Variable(start_tensor)
    end_label = Variable(end_tensor)

    start, end, prediction = self.qa_module(instance)
    # noinspection PyTypeChecker
    loss = F.cross_entropy(start, start_label) + F.cross_entropy(end, end_label)
    self.model_loss.calculate_loss(loss)

    loss.backward()

    # Clip gradients to avoid gradients explosion
    # TODO: Move max_norm value to config
    max_norm = 5
    utils.clip_grad_norm_(self.qa_module.parameters(), max_norm)

    # Calculate parameters with optimizer step
    self.optimizer.step()

  def get_tensor(self, tensor) -> Tensor:
    """

    :rtype: Tensor
    """
    return tensor.cuda() if self.cuda_on else tensor

  def predict(self):
    self.qa_module.eval()
    pass

  def eval(self):
    pass

  def save(self):
    pass