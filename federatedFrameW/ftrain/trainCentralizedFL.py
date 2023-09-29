import copy
import queue
import re
import time
from abc import abstractmethod

import h5py
import numpy as np
import torch
# from aim import Run
from torch import nn

from federatedFrameW.base.ftrainbase import tbase
from federatedFrameW.utils.data_utils import read_data, read_user_data
from federatedFrameW.utils.loss_utils import get_loss
from federatedFrameW.utils.plot_utils import get_hparams_name, simple_read_data


# def save_aim(file_name, train_acc=[], train_loss=[], glob_acc=[], test_loss=[]):
#     hp = get_hparams_name(file_name)
#     run = Run(experiment=file_name)
#     run["hparams"] = hp
#     for i in range(len(train_acc)):
#         run.track(train_acc[i], 'train_acc', epoch=i)
#         run.track(train_loss[i], 'train_loss', epoch=i)
#         run.track(glob_acc[i], 'glob_acc', epoch=i)
#         run.track(test_loss[i], 'test_loss', epoch=i)


class CentralizedFL(tbase):
    '''
    This class is used to train a centralized federated learning model.

    kwargs:
        - dataset: the name of the dataset
        - device: the device to train the model on
        - model: the origin model for deepcopy
        - name: the name of the algorithm
        - lImp: the local implementation
        - gImp: the global implementation
        - nImp: the node implementation
        - hyperparams: the hyperparameters
            - batch_size: batch size
            - total_epochs: total number of epochs
            - local_epochs: int, number of epochs for local training
            - beta: global momentum
            - num_aggregate_locals: number of local models to aggregate
            - learning_rate: learning rate

    abstract methods:
        - res_file_name: results file name
    '''

    def __init__(self, *args, **kwargs):
        self.dataset = kwargs['dataset']
        self.device = kwargs['device']
        self.model = kwargs['model']
        self.name = kwargs['name']
        self.lImp = kwargs['lImp']
        self.gImp = kwargs['gImp']
        self.nImp = kwargs['nImp']
        super().__init__(*args, **kwargs)

        self.test_topk = 10
        self.lastest_eval_l_teacc = queue.Queue(maxsize=self.test_topk)
        self.lastest_eval_l_tracc = queue.Queue(maxsize=self.test_topk)

    def gen_locals(self):
        data = read_data(self.dataset)
        self.total_users = len(data[0])
        for i in range(self.total_users):
            id, train_data, test_data = read_user_data(i, data, self.dataset, self.hyperparams)
            model = copy.deepcopy(self.model)
            hyperparams = copy.deepcopy(self.hyperparams)
            local = self.lImp(id=id, device=self.device, model=model, train_data=train_data, test_data=test_data,
                              hyperparams=hyperparams)
            self.flocals.append(local)
            self.total_train_samples += local.train_samples

    def gen_globals(self):
        model = copy.deepcopy(self.model)
        hyperparams = copy.deepcopy(self.hyperparams)
        self.fglobals.append(self.gImp(id=0, model=model, hyperparams=hyperparams))

    def gen_nodes(self):
        node = self.nImp(id=0, fglobals=self.fglobals, flocals=self.flocals, hyperparams=self.hyperparams)
        self.fnodes.append(node)

    def gen_actions(self):
        for e in range(self.total_epochs):
            self.actions.append(self.local_train)
            self.actions.append(self.global_aggregate)
            self.actions.append(self.global_eval)
            self.actions.append(self.local_eval)

    @abstractmethod
    def res_file_name(self, tag=''):
        pass

    def save_global_results(self):
        file_name = self.res_file_name(tag='_g')

        for fn in self.fnodes:
            for fg in fn.fglobals:
                if ((len(fg.rs_glob_acc) != 0)
                        & (len(fg.rs_train_acc) != 0)
                        & (len(fg.rs_train_loss) != 0)
                        & (len(fg.rs_test_loss) != 0)):
                    print("results/" + '{}.h5'.format('[' + str(fn.id) + '-' + str(fg.id) + ']' + file_name))
                    # with h5py.File("results/" + '{}.h5'.format('[' + str(fn.id) + '-' + str(fg.id) + ']' + file_name),
                    #                'w') as hf:
                    #     hf.create_dataset('rs_glob_acc', data=fg.rs_glob_acc)
                    #     hf.create_dataset('rs_train_acc', data=fg.rs_train_acc)
                    #     hf.create_dataset('rs_train_loss', data=fg.rs_train_loss)
                    #     hf.create_dataset('rs_test_loss', data=fg.rs_test_loss)
                    #     hf.close()

                    fg.save_mdl2p(fn.id, self.res_file_name())

                    # save_aim(re.split('_\d+$', file_name)[0], train_acc=fg.rs_glob_acc, train_loss=fg.rs_train_loss,
                    #          glob_acc=fg.rs_glob_acc,
                    #          test_loss=fg.rs_test_loss)

    def local_train(self):
        self.pre_local_train()
        self.train_local_model()

    def global_aggregate(self):
        self.pre_global_aggregate()
        self.aggregate_global_model()

    def local_eval(self):
        self.pre_local_eval()
        self.eval_local_model()

    def global_eval(self):
        self.pre_global_eval()
        self.eval_global_model()

    def pre_train(self):
        self.c = 0

    def pre_local_train(self):
        self.t = time.time()
        self.c += 1
        print("-------------Round number: {0:05d}".format(self.c), " -------------")
        for fn in self.fnodes:
            fn.sample_local()

    def train_local_model(self):
        for fn in self.fnodes:
            fn.train()
        print("--------------LT time: {:.04f}".format(time.time() - self.t), " ----------------")

    def post_train(self):
        self.save_global_results()

    def pre_global_aggregate(self):
        pass

    def aggregate_global_model(self):
        for fn in self.fnodes:
            fn.aggregate_global()

    def pre_local_eval(self):
        pass

    def eval_local_model(self):
        pass

    def pre_global_eval(self):
        self.t = time.time()

    def eval_global_model(self):
        ler = []
        for fn in self.fnodes:
            ler.append(fn.eval_global())
        print("--------------EG time: {:.04f}".format(time.time() - self.t), " ----------------")
        # self.lastest_eval_l_teacc.put(ler)
        # self.lastest_eval_l_tracc.put(ler)

    def pre_action(self, action):
        pass

    def post_action(self, action):
        pass
