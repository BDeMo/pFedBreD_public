from Algorithms.FedAvg.globalFedAvg import gFedAvg
from Algorithms.FedAvg.localFedAvg import lFedAvg
from Algorithms.FedAvg.nodeFedAvg import nFedAvg
from federatedFrameW.ftrain.trainCentralizedFL import CentralizedFL
from federatedFrameW.utils.model_utils import general_CenFL_filename


class FedAvg(CentralizedFL):
    '''
    Federated Learning with FedAvg

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
            - times: the number of times to repeat the experiment
            - loss_name: the name of the loss function
            - optimizer_name: the name of the optimizer
    '''

    def __init__(self, *args, **kwargs):
        kwargs['lImp'] = lFedAvg
        kwargs['gImp'] = gFedAvg
        kwargs['nImp'] = nFedAvg
        super().__init__(*args, **kwargs)

    def res_file_name(self, tag=''):
        return general_CenFL_filename(self, tag) \
               + "_" + str(self.times)
