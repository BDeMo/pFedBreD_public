import queue
import re
import time
from abc import abstractmethod

import h5py
import torch

# from federatedFrameW.ftrain.trainCentralizedFL import save_aim
from federatedFrameW.ftrain.trainCentralizedFL import CentralizedFL
from federatedFrameW.utils.plot_utils import get_hparams_name


class CentralizedPerFL(CentralizedFL):
    '''
    Centralized Personalized Federated Learning

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
            - personal_learning_rate: personalized model learning rate
            - times: the number of times to repeat the experiment

    abstract methods:
        - res_file_name: results file name
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.lastest_eval_g_teacc = queue.Queue(maxsize=self.test_topk)
        # self.lastest_eval_g_tracc = queue.Queue(maxsize=self.test_topk)

    @abstractmethod
    def res_file_name(self, tag=''):
        pass

    def save_local_results(self):
        file_name = self.res_file_name(tag='_p')

        for fn in self.fnodes:
            for fg in fn.fglobals:
                if ((len(fg.rs_glob_acc_per) != 0)
                        & (len(fg.rs_train_acc_per) != 0)
                        & (len(fg.rs_train_loss_per) != 0)
                        & (len(fg.rs_test_loss_per) != 0)):
                    print("results/" + '{}.h5'.format('[' + str(fn.id) + '-' + str(fg.id) + ']' + file_name))
                    # with h5py.File(
                    #         "results/" + '{}.h5'.format('[' + str(fn.id) + '-' + str(fg.id) + ']' + file_name),
                    #         'w') as hf:
                    #     hf.create_dataset('rs_glob_acc', data=fg.rs_glob_acc_per)
                    #     hf.create_dataset('rs_train_acc', data=fg.rs_train_acc_per)
                    #     hf.create_dataset('rs_train_loss', data=fg.rs_train_loss_per)
                    #     hf.create_dataset('rs_test_loss', data=fg.rs_test_loss_per)
                    #     hf.close()

                    for fl in fg.candidate_locals:
                        fl.save_mdl2p(fn.id, fg.id, self.res_file_name())

                    # save_aim(re.split('_\d+$', file_name)[0], train_acc=fg.rs_train_acc_per,
                    #          train_loss=fg.rs_train_loss_per,
                    #          glob_acc=fg.rs_glob_acc_per,
                    #          test_loss=fg.rs_test_loss_per)

    def post_train(self):
        self.save_global_results()
        self.save_local_results()

    def pre_local_eval(self):
        self.t = time.time()

    def eval_local_model(self):
        ger = []
        for fn in self.fnodes:
            ger.append(fn.eval_local())
        print("==============EL time: {:.04f}".format(time.time() - self.t), " =================")
        # self.lastest_eval_g_teacc.put(ger)
        # self.lastest_eval_g_tracc.put(ger)


if __name__ == '__main__':
    file_name = 'sent140_pFedBreD_ns_mg_lstm_NLLLoss_p_0.01lr_20nal_20bs_1.0b_20le_0.01pl_95.0lam_5pi_0.05eta_0'
    print(re.split('_\d+$', file_name)[0])
    print(get_hparams_name(re.split('_\d+$', file_name)[0]))
