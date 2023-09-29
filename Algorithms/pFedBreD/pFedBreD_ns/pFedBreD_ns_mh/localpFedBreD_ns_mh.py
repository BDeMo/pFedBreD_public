import copy

from torch.nn.utils import parameters_to_vector as P2V, vector_to_parameters as V2P

from federatedFrameW.base.flocalbase import lbase
from federatedFrameW.utils.loss_utils import get_loss
from federatedFrameW.utils.optim_utils import get_optimizer


class lpFedBreD_ns_mh(lbase):
    '''
    local Class for pFedBreD_ns_mh

    kwargs:
        - id: local id
        - device: device
        - model: calculation model deepcopy
        - hyperparams:
            - batch_size: batch size
            - local_epochs: int, number of epochs for local training
            - optimizer_name: str, name of optimizer
            - loss: loss function
            - prox_iters: int, number of proximal solution iterations
            - eta: extra parameter for proximal solution
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.current_grad = copy.deepcopy(list(self.model.parameters()))
        self.last_model = copy.deepcopy(list(self.model.parameters()))
        self.s_model = copy.deepcopy(list(self.model.parameters()))
        self.alpha_optimizer = get_optimizer('SGD_lrs')(self.model.parameters(), self.hyperparams)
        self.c = 0

    def gen_loss(self):
        return get_loss(self.loss_name)()

    def gen_optimizer(self):
        # return get_optimizer('ns_iter_Optimizer')(self.model.parameters(), self.hyperparams)
        return get_optimizer('ns_Optimizer')(self.model.parameters(), self.hyperparams)

    def train(self):
        LOSS = 0
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            eta = self.eta
            eta_alpha = 1e-2

            temp_model = copy.deepcopy(list(self.model.parameters()))
            X, y = self.get_next_train_batch()
            self.alpha_optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.alpha_optimizer.step(eta=eta_alpha)

            X, y = self.get_next_train_batch()
            self.alpha_optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.set_model_parameters(temp_model)
            self.alpha_optimizer.step(eta=eta_alpha)
            self.clone_model_paramenter(self.model.parameters(), self.local_model)

            self.s_model = copy.deepcopy(self.last_model)
            for lstm_param, lm_param, pm_param, sm_param in zip(self.last_model, self.local_model,
                                                                self.personalized_model, self.s_model):
                if lm_param.requires_grad:
                    sm_param.data = lm_param.data - eta * (lstm_param.data - pm_param.data)
            for i in range(self.prox_iters):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step(self.s_model)
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model)
            for lm_param, pm_param, sm_param in zip(self.local_model,
                                                    self.personalized_model, self.s_model):
                if lm_param.requires_grad:
                    lm_param.data = lm_param.data - self.learning_rate * self.lamda * (sm_param.data - pm_param.data)
            self.set_model_parameters(self.local_model)
        self.last_model = copy.deepcopy(self.local_model)

        return LOSS
