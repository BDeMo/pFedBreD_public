import time

from Algorithms.PerFedAvg.globalPerFedAvg import gPerFedAvg
from Algorithms.PerFedAvg.localPerFedAvg import lPerFedAvg
from Algorithms.PerFedAvg.nodePerFedAvg import nPerFedAvg
from federatedFrameW.ftrain.trainCentralizedPerFL import CentralizedPerFL
from federatedFrameW.utils.model_utils import general_CenPerFL_filename


class PerFedAvg(CentralizedPerFL):
    '''
    Federated Learning with PerFedAvg

    kwargs:
        - dataset: the name of the dataset
        - device: the device to train the model on
        - model: the origin model for deepcopy
        - name: the name of the algorithm
        - lImp: the local implementation
        - gImp: the global implementation
        - nImp: the node implementation
        - hyperparams: the hyperparameters
            - model_name: the name of the model
            - batch_size: batch size
            - total_epochs: total number of epochs
            - local_epochs: int, number of epochs for local training
            - beta: global momentum
            - num_aggregate_locals: number of local models to aggregate
            - learning_rate: learning rate
            - personal_learning_rate: personalized model learning rate
            - times: the number of times to repeat the experiment
            - optimizer_name: the name of optimizer for locals
            - loss: the loss function for locals
            - loss_name: the name of the loss function
    '''

    def __init__(self, *args, **kwargs):
        kwargs['lImp'] = lPerFedAvg
        kwargs['gImp'] = gPerFedAvg
        kwargs['nImp'] = nPerFedAvg
        super().__init__(*args, **kwargs)

    def res_file_name(self, tag=''):
        return general_CenPerFL_filename(self, tag) \
               + "_" + str(self.times)

    def pre_local_eval(self):
        for fn in self.fnodes:
            fn.gen_personalized_model()
        self.t = time.time()
