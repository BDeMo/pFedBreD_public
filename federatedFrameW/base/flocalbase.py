import os
from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

from federatedFrameW.utils.loss_utils import get_loss
from federatedFrameW.utils.optim_utils import get_optimizer


class lbase(metaclass=ABCMeta):
    """
    Abstract local trainning and local data processing base class

    kwargs:
        - id: local id
        - device: device
        - model: calculation model deepcopy
        - hyperparams:
            - batch_size: batch size

    abstract methods:
        - train: local trainning
        - gen_loss: generate loss
        - gen_optimizer: generate optimizer
    """

    def __init__(self, *args, **kwargs):
        self.id = kwargs['id']
        self.device = kwargs['device']
        self.model = kwargs['model']
        self.hyperparams = kwargs['hyperparams']
        if len(self.hyperparams) > 0:
            self.process_hyperparams()

        self.load_data(kwargs['train_data'], kwargs['test_data'], self.batch_size)
        self.loss = self.gen_loss()
        self.optimizer = self.gen_optimizer()
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model = copy.deepcopy(list(self.model.parameters()))
        self.mdl_param = {'global_acc': copy.deepcopy(list(self.model.parameters()))
            , 'train_acc': copy.deepcopy(list(self.model.parameters()))
            , 'test_loss': copy.deepcopy(list(self.model.parameters()))
            , 'train_loss': copy.deepcopy(list(self.model.parameters()))
                          }

    @abstractmethod
    def gen_loss(self):
        pass

    @abstractmethod
    def gen_optimizer(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def save_mdl(self, names):
        for name in names:
            self.clone_model_paramenter(self.personalized_model, self.mdl_param[name])

    def load_data(self, train_data, test_data, batch_size):
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.trainloader = DataLoader(train_data, batch_size, shuffle=True)
        self.testloader = DataLoader(test_data, batch_size, shuffle=True)
        self.trainloader_full = DataLoader(train_data, len(train_data))
        self.testloader_full = DataLoader(test_data, len(test_data))
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

    def process_hyperparams(self):
        for hp in self.hyperparams.keys():
            setattr(self, hp, self.hyperparams[hp])

    def set_model_parameters(self, model):
        if isinstance(model, nn.Parameter):
            for old_param, new_param in zip(self.model.parameters(), model.parameters()):
                old_param.data = new_param.data.clone()
        elif isinstance(model, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = model[idx].clone()
        else:
            raise Exception('Setting model parameters, un proper tyep error')

    def set_local_parameters(self, model):
        if isinstance(model, nn.Parameter):
            for old_param, new_param in zip(self.local_model, model.parameters()):
                old_param.data = new_param.data.clone()
        elif isinstance(model, list):
            for idx, model_grad in enumerate(self.local_model):
                model_grad.data = model[idx].clone()
        else:
            raise Exception('Setting local model parameters, un proper tyep error')

    def get_model_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param_, clone_param_ in zip(param, clone_param):
            clone_param_.data = param_.data.clone()
        return clone_param

    def save_mdl2p(self, nid, gid, file_name):
        for name in self.mdl_param.keys():
            if not os.path.isdir('mdl_param/' + file_name):
                os.mkdir('mdl_param/' + file_name)
            if not os.path.isdir('mdl_param/' + file_name):
                os.mkdir('mdl_param/' + file_name)
                os.mkdir('mdl_param/' + file_name + '/' + name)
            elif not os.path.isdir('mdl_param/' + file_name + '/' + name):
                os.mkdir('mdl_param/' + file_name + '/' + name)
            path = 'mdl_param/' + file_name + '/' + name + '/' + '[' + str(nid) + '-' + str(gid) + '_' + str(
                self.id) + ']' + '.mdl'
            torch.save(self.mdl_param[name], path)

    def test(self):
        self.model.eval()
        test_acc = 0
        test_loss = 0
        # for p, l in zip(self.personalized_model, list(self.model.parameters())):
        #     print('test:\t', p - l)
        for x, y in self.testloader_full:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            # print('pred:', (torch.sum(torch.argmax(output, dim=1)) / output.shape[0]).item())
            # print('y:', (torch.sum(y) / y.shape[0]).item())
            if len(y.shape) > 1 and y.shape[1] > 1:
                test_acc += (torch.sum(torch.argmax(output, dim=1) == torch.argmax(y, dim=1))).item()
            else:
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            test_loss += self.loss(output, y)
        return test_acc, test_loss, self.test_samples

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        # for p, l in zip(self.personalized_model, list(self.model.parameters())):
        #     print('tl:\t', p - l)
        for x, y in self.trainloader_full:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            if len(y.shape) > 1 and y.shape[1] > 1:
                train_acc += (torch.sum(torch.argmax(output, dim=1) == torch.argmax(y, dim=1))).item()
            else:
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return train_acc, loss, self.train_samples

    def test_personalized_model(self):
        self.model.eval()
        test_acc = 0
        test_loss = 0
        self.set_model_parameters(self.personalized_model)
        # for p, l in zip(self.personalized_model, list(self.model.parameters())):
        #     print('test_pl:\t', p - l)
        for x, y in self.testloader_full:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            if len(y.shape) > 1 and y.shape[1] > 1:
                test_acc += (torch.sum(torch.argmax(output, dim=1) == torch.argmax(y, dim=1))).item()
            else:
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            test_loss += self.loss(output, y)
        return test_acc, test_loss, self.test_samples

    def train_error_and_loss_personalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.set_model_parameters(self.personalized_model)
        # for p, l in zip(self.personalized_model, list(self.model.parameters())):
        #     print('tl_pl:\t', p - l)
        for x, y in self.trainloader_full:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            if len(y.shape) > 1 and y.shape[1] > 1:
                train_acc += (torch.sum(torch.argmax(output, dim=1) == torch.argmax(y, dim=1))).item()
            else:
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return train_acc, loss, self.train_samples

    def get_next_train_batch(self):
        try:
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))

    def get_next_train_batch_full(self):
        try:
            (X, y) = next(iter(self.trainloader_full))
        except StopIteration:
            (X, y) = next(iter(self.trainloader_full))
        return (X.to(self.device), y.to(self.device))

    def get_next_test_batch(self):
        try:
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))
