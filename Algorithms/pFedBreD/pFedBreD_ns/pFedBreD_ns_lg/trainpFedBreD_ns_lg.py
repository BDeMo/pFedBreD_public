from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_lg.globalpFedBreD_ns_lg import gpFedBreD_ns_lg
from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_lg.localpFedBreD_ns_lg import lpFedBreD_ns_lg
from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_lg.nodepFedBreD_ns_lg import npFedBreD_ns_lg
from federatedFrameW.ftrain.trainCentralizedPerFL import CentralizedPerFL
from federatedFrameW.utils.model_utils import general_CenPerFL_RegCoeff_filename


class pFedBreD_ns_lg(CentralizedPerFL):
    '''
    Personalized Federated Learning with pFedBreD_ns_lg

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
            - lamda: lambda in pFedMe reg-coeff
            - num_aggregate_locals: number of local models to aggregate
            - learning_rate: learning rate
            - personal_learning_rate: personalized model learning rate
            - times: the number of times to repeat the experiment
            - loss_name: the name of the loss function
            - optimizer_name: the name of the optimizer
            - prox_iters: int, number of proximal solution iterations
            - eta: float, the parameter for the proximal solution
    '''

    def __init__(self, *args, **kwargs):
        kwargs['lImp'] = lpFedBreD_ns_lg
        kwargs['gImp'] = gpFedBreD_ns_lg
        kwargs['nImp'] = npFedBreD_ns_lg
        super().__init__(*args, **kwargs)

    def res_file_name(self, tag=''):
        return general_CenPerFL_RegCoeff_filename(self, tag) \
               + "_" + str(self.eta) + "eta" \
               + "_" + str(self.times)
