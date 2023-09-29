import argparse
import random

import numpy as np
import torch

from Algorithms.FedAvg.trainFedAvg import FedAvg
from Algorithms.PerFedAvg.trainPerFedAvg import PerFedAvg
from Algorithms.pFedMe.trainpFedMe import pFedMe
from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_lg.trainpFedBreD_ns_lg import pFedBreD_ns_lg
from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_mh.trainpFedBreD_ns_mh import pFedBreD_ns_mh
from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_meg.trainpFedBreD_ns_meg import pFedBreD_ns_meg
# from Algorithms.pFedBreD.pFedBreD_kl.pFedBreD_kl_fo.trainpFedBreD_kl_fo import pFedBreD_kl_fo
# from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_fm.trainpFedBreD_ns_fm import pFedBreD_ns_fm
# from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_fmd.trainpFedBreD_ns_fmd import pFedBreD_ns_fmd
# from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_meg_ft.trainpFedBreD_ns_meg_ft import pFedBreD_ns_meg_ft
# from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_mh_ft.trainpFedBreD_ns_mh_ft import pFedBreD_ns_mh_ft
# from Algorithms.pFedBreD.pFedBreD_ns.pFedBreD_ns_lg_ft.trainppFedBreD_ns_lg_ft import pFedBreD_ns_lg_ft
# from Algorithms.FedEM.trainFedEM import FedEM
# from Algorithms.FedEM_ft.trainFedEM_ft import FedEM_ft
# from Algorithms.PerFedAvg_ft.trainPerFedAvg_ft import PerFedAvg_ft
# from Algorithms.pFedBayes.trainFedBayes import pFedBayes
# from Algorithms.FedAvg.trainFedHN import FedHN
# from Algorithms.PerFedAvg.trainFedPAC import FedPAC
# from Algorithms.FedAvg.trainDitto import Ditto
# from Algorithms.PerFedAvg.trainFedfomo import Fedfomo
# from Algorithms.FedAMP.trainFedAMP import FedAMP
# from Algorithms.pFedMe_ft.trainpFedMe_ft import pFedMe_ft

from federatedFrameW.models.models import Mclr_Logistic, DNN, CifarNet, Sent140_LSTM, Shkspr_LSTM, \
    Mclr_Logistic_Femnist, DNN_Femnist

SEED = 2022
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True


def main(*args, **kwargs):
    '''
    kwargs:
        - gpu: the device to train the model on
        - dataset: the name of the dataset
        - name: the name of the algorithm
        - model_name: the origin model for deepcopy
        - loss_name: the loss function to use
        - optimizer_name: the optimizer to use
        - batch_size: batch size
        - total_epochs: total number of epochs
        - local_epochs: int, number of epochs for local training
        - beta: global momentum
        - num_aggregate_locals: number of local models to aggregate
        - learning_rate: learning rate
        - personal_learning_rate: personal learning rate
        - times: the number of times to repeat the experiment
        - eta: the extra parameter
    '''
    dataset = kwargs['dataset']
    name = kwargs['name']
    model_name = kwargs['model_name']
    loss_name = kwargs['loss_name']
    optimizer_name = kwargs['optimizer_name']
    batch_size = kwargs['batch_size']
    learning_rate = kwargs['learning_rate']
    beta = kwargs['beta']
    lamda = kwargs['lamda']
    total_epochs = kwargs['total_epochs']
    local_epochs = kwargs['local_epochs']
    num_aggregate_locals = kwargs['num_aggregate_locals']
    prox_iters = kwargs['prox_iters']
    personal_learning_rate = kwargs['personal_learning_rate']
    times = kwargs['times']
    gpu = kwargs['gpu']
    eta = kwargs['eta']
    tau = kwargs['tau']

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    hyperparams = {
        'model_name': model_name
        , 'loss_name': loss_name
        , 'name': name
        , 'total_epochs': total_epochs
        # general FL
        , 'learning_rate': learning_rate
        , 'num_aggregate_locals': num_aggregate_locals
        , 'batch_size': batch_size
        , 'beta': beta
        , 'local_epochs': local_epochs
        , 'optimizer_name': optimizer_name
        , 'times': times
        # general PerFL
        , 'personal_learning_rate': personal_learning_rate
        # general PerFL with Reg or Bi-Level Optim
        , 'lamda': lamda
        , 'prox_iters': prox_iters
        , 'eta': eta
        , 'tau': tau
    }

    tModelList = {
        ('mclr', 'mnist'): Mclr_Logistic
        , ('dnn', 'mnist'): DNN
        , ('mclr', 'femnist'): Mclr_Logistic_Femnist
        , ('dnn', 'femnist'): DNN_Femnist
        , ('mclr', 'fashion_mnist'): Mclr_Logistic
        , ('dnn', 'fashion_mnist'): DNN
        , ('cnn', 'cifar10'): CifarNet
        , ('lstm', 'sent140'): Sent140_LSTM
        , ('lstm', 'shakespeare'): Shkspr_LSTM
    }

    model = tModelList[(model_name, dataset)]().to(device)

    tClassList = {
        'FedAvg': FedAvg
        , 'PerFedAvg': PerFedAvg
        , 'pFedMe': pFedMe
        , 'pFedBreD_ns_lg': pFedBreD_ns_lg
        # , 'pFedBreD_ns_fm': pFedBreD_ns_fm
        # , 'pFedBreD_ns_fmd': pFedBreD_ns_fmd
        , 'pFedBreD_ns_mh': pFedBreD_ns_mh
        , 'pFedBreD_ns_meg': pFedBreD_ns_meg
        # , 'FedHN': FedHN
        # , 'FedPAC': FedPAC
        # , 'Fedfomo': Fedfomo
        # , 'Ditto': Ditto
        # , 'FedAMP': FedAMP
        # , 'FedAMP_ft': FedAMP_ft
        # , 'FedEM': FedEM
        # , 'FedEM_ft': FedEM_ft
        # , 'pFedBayes': pFedBayes
    }
    for t in range(times):
        hyperparams['times'] = t
        trainer = tClassList[name](device=device, name=name, model=model, dataset=dataset, hyperparams=hyperparams)
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashion_mnist",
                        choices=["mnist", "femnist", "synthetic", "cifar10", "fashion_mnist", "sent140", "shakespeare"])
    parser.add_argument("--model_name", type=str, default="mclr"
                        , choices=["dnn", "mclr", "cnn", "lstm"])
    parser.add_argument("--loss_name", type=str, default="NLLLoss"
                        , choices=["NLLLoss", "CrossEntropyLoss", "BCEWithLogitsLoss", "BCELoss"])
    parser.add_argument("--optimizer_name", type=str, default="SGD"
                        , choices=["SGD", "Adam", "Adagrad", "pFedMeOptimizer"])
    parser.add_argument("--name", type=str, default="pFedBreD_ns_meg"
                        , choices=["pFedMe", "pFedMe_ft", "PerFedAvg", "PerFedAvg_ft", "FedAvg", "FedAMP", "FedAMP_ft", "pFedBayes", "FedEM", "FedEM_ft", "pFedBreD_ns_fm", "pFedBreD_ns_fmd",
                                   "pFedBreD_ns_lg", "pFedBreD_ns_mh", "pFedBreD_ns_meg", "pFedBreD_kl_fo", "FedHN", "FedPAC", "Fedfomo", "Ditto"])
    parser.add_argument("--batch_size", type=int, default=20
                        , help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-2
                        , help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0
                        , help="Average moving")
    parser.add_argument("--lamda", type=float, default=15
                        , help="Regularization term")
    parser.add_argument("--total_epochs", type=int, default=1
                        , help="Total global iteration")
    parser.add_argument("--local_epochs", type=int, default=20
                        , help="Local iteration between aggregation")
    parser.add_argument("--num_aggregate_locals", type=int, default=20
                        , help="Number of Users per round")
    parser.add_argument("--prox_iters", type=int, default=5
                        , help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=1e-2
                        , help="Personalized learning rate")
    parser.add_argument("--times", type=int, default=1
                        , help="Running time")
    parser.add_argument("--gpu", type=int, default=0
                        , help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--eta", type=float, default=5e-2
                        , help="Extra hyperparam")
    parser.add_argument("--tau", type=float, default=1e-2
                        , help="Extra hyperparam")
    args = parser.parse_args()

    kwargs = {
        'gpu': args.gpu
        , 'name': args.name
        , 'dataset': args.dataset
        , 'model_name': args.model_name
        , 'loss_name': args.loss_name
        , 'optimizer_name': args.optimizer_name
        , 'total_epochs': args.total_epochs
        , 'local_epochs': args.local_epochs
        , 'learning_rate': args.learning_rate
        , 'num_aggregate_locals': args.num_aggregate_locals
        , 'batch_size': args.batch_size
        , 'beta': args.beta
        , 'personal_learning_rate': args.personal_learning_rate
        , 'lamda': args.lamda
        , 'prox_iters': args.prox_iters
        , 'times': args.times
        , 'eta': args.eta
        , 'tau': args.tau
    }

    print("=" * 80)
    for k in kwargs.keys():
        print(k + ':\t{}'.format(kwargs[k]))
    print("=" * 80)

    main(**kwargs)
