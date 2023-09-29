import os.path
from abc import abstractmethod, ABCMeta

import torch
import numpy as np
import copy


class gbase(metaclass=ABCMeta):
    """
    Abstract global communication and management class

    kwargs:
        - id: id of the global
        - model: calculation model deepcopy
        - hyperparame:
            - beta: global momentum

    abstract methods:
        - sample_locals: sample local calculations
    """

    def __init__(self, *args, **kwargs):
        self.id = kwargs['id']
        self.model = kwargs['model']
        self.hyperparams = kwargs['hyperparams']
        if len(self.hyperparams) > 0:
            self.process_hyperparams()

        self.candidate_locals = []
        self.sampled_locals = []
        self.rs_test_loss, \
        self.rs_train_acc, \
        self.rs_train_loss, \
        self.rs_glob_acc, \
        self.rs_test_loss_per, \
        self.rs_train_acc_per, \
        self.rs_train_loss_per, \
        self.rs_glob_acc_per \
            = [], [], [], [], [], [], [], []
        self.mdl_param = {'global_acc': copy.deepcopy(list(self.model.parameters()))
            , 'train_acc': copy.deepcopy(list(self.model.parameters()))
            , 'test_loss': copy.deepcopy(list(self.model.parameters()))
            , 'train_loss': copy.deepcopy(list(self.model.parameters()))
                          }

    def save_mdl2p(self, nid, file_name):
        for name in self.mdl_param.keys():
            if not os.path.isdir('mdl_param/' + file_name):
                os.mkdir('mdl_param/' + file_name)
                os.mkdir('mdl_param/' + file_name + '/' + name)
            elif not os.path.isdir('mdl_param/' + file_name + '/' + name):
                os.mkdir('mdl_param/' + file_name + '/' + name)
            path = 'mdl_param/' + file_name + '/' + name + '/' + '[' + str(nid) + '-' + str(self.id) + ']' + '.mdl'
            torch.save(self.mdl_param[name], path)

    def process_hyperparams(self):
        for hp in self.hyperparams.keys():
            setattr(self, hp, self.hyperparams[hp])

    @abstractmethod
    def sample_locals(self):
        pass

    def clone_model_paramenter(self, param, clone_param):
        for param_, clone_param_ in zip(param, clone_param):
            clone_param_.data = param_.data.clone()
        return clone_param

    def send_parameters(self):
        assert (self.candidate_locals is not None and len(self.candidate_locals) > 0)
        for local in self.candidate_locals:
            local.set_model_parameters(list(self.model.parameters()))
            local.set_local_parameters(list(self.model.parameters()))

    def send_parameters_sampled_locals(self):
        assert (self.sampled_locals is not None and len(self.sampled_locals) > 0)
        for local in self.sampled_locals:
            local.set_model_parameters(list(self.model.parameters()))
            local.set_local_parameters(list(self.model.parameters()))

    def add_parameters(self, local, ratio):
        for server_param, local_param in zip(self.model.parameters(), local.local_model):
            server_param.data = server_param.data + local_param.data.clone() * ratio

    def aggregate_parameters(self, beta=1):
        assert (self.candidate_locals is not None and len(self.candidate_locals) > 0)

        # store previous parameters
        if beta != 1:
            previous_param = copy.deepcopy(list(self.model.parameters()))

        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for local in self.sampled_locals:
            total_train += local.train_samples

        for local in self.sampled_locals:
            self.add_parameters(local, local.train_samples / total_train)

        # aggregate avergage trainmodel with previous trainmodel using parameter beta
        if beta != 1:
            for param, pre_param in zip(self.model.parameters(), previous_param):
                param.data = (1 - beta) * pre_param.data + beta * param.data

    def select_locals_uniform(self, num_aggregate_locals):
        if (num_aggregate_locals == len(self.candidate_locals)):
            return self.candidate_locals

        num_aggregate_locals = min(num_aggregate_locals, len(self.candidate_locals))
        # np.random.seed(round)
        return np.random.choice(self.candidate_locals, num_aggregate_locals, replace=False)  # , p=pk)

    def test(self):
        num_samples = []
        test_loss = []
        tot_correct = []
        for c in self.candidate_locals:
            ct, tl, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            test_loss.append(tl)
        ids = [c.id for c in self.candidate_locals]

        return ids, num_samples, tot_correct, test_loss

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.candidate_locals:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.candidate_locals]

        return ids, num_samples, tot_correct, losses

    def test_personalized_model(self):
        num_samples = []
        test_loss = []
        tot_correct = []
        for c in self.candidate_locals:
            ct, tl, ns = c.test_personalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            test_loss.append(tl)
        ids = [c.id for c in self.candidate_locals]

        return ids, num_samples, tot_correct, test_loss

    def train_error_and_loss_personalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.candidate_locals:
            ct, cl, ns = c.train_error_and_loss_personalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.candidate_locals]

        return ids, num_samples, tot_correct, losses

    def save_mdl_g(self, names):
        for name in names:
            self.clone_model_paramenter(self.model.parameters(), self.mdl_param[name])

    def save_mdl_p(self, names):
        for c in self.candidate_locals:
            c.save_mdl(names)

    def evaluate(self):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        test_loss = sum([x * y for (x, y) in zip(stats[1], stats[3])]).item() / np.sum(stats[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])

        l2sv = []
        if len(self.rs_glob_acc) > 0 and glob_acc > max(self.rs_glob_acc):
            l2sv.append('global_acc')
        if len(self.rs_train_acc) > 0 and train_acc > max(self.rs_train_acc):
            l2sv.append('train_acc')
        if len(self.rs_test_loss) > 0 and test_loss < min(self.rs_test_loss):
            l2sv.append('test_loss')
        if len(self.rs_train_loss) > 0 and train_loss < min(self.rs_train_loss):
            l2sv.append('train_loss')

        self.save_mdl_g(l2sv)

        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_test_loss.append(test_loss)
        self.rs_train_loss.append(train_loss)
        print(self.name + "-" + "Global Testing Accurancy: ", glob_acc)
        print(self.name + "-" + "Global Trainning Accurancy: ", train_acc)
        print(self.name + "-" + "Global Testing Loss: ", test_loss)
        print(self.name + "-" + "Global Trainning Loss: ", train_loss)
        return glob_acc, train_acc, train_loss

    def evaluate_personalized_model(self):
        stats = self.test_personalized_model()
        stats_train = self.train_error_and_loss_personalized_model()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        test_loss = sum([x * y for (x, y) in zip(stats[1], stats[3])]).item() / np.sum(stats[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])

        l2sv = []
        if len(self.rs_glob_acc) > 0 and glob_acc > max(self.rs_glob_acc):
            l2sv.append('global_acc')
        if len(self.rs_train_acc) > 0 and train_acc > max(self.rs_train_acc):
            l2sv.append('train_acc')
        if len(self.rs_test_loss) > 0 and test_loss < min(self.rs_test_loss):
            l2sv.append('test_loss')
        if len(self.rs_train_loss) > 0 and train_loss < min(self.rs_train_loss):
            l2sv.append('train_loss')

        self.save_mdl_p(l2sv)

        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_test_loss_per.append(test_loss)
        self.rs_train_loss_per.append(train_loss)
        print(self.name + "-" + "Average Personal Testing Accurancy: ", glob_acc)
        print(self.name + "-" + "Average Personal Trainning Accurancy: ", train_acc)
        print(self.name + "-" + "Average Personal Testing Loss: ", test_loss)
        print(self.name + "-" + "Average Personal Trainning Loss: ", train_loss)
        return glob_acc, train_acc, train_loss
