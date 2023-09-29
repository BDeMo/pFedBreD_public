import copy

from torch.nn.utils import parameters_to_vector as P2V, vector_to_parameters as V2P

from federatedFrameW.base.flocalbase import lbase
from federatedFrameW.utils.loss_utils import get_loss
from federatedFrameW.utils.optim_utils import get_optimizer


class lpFedBreD_ns_lg(lbase):
    '''
    local Class for pFedBreD_ns_lg

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
        self.s_model = copy.deepcopy(list(self.model.parameters()))
        self.alpha_optimizer = get_optimizer('SGD_lrs')(self.model.parameters(), self.hyperparams)
        self.c = 0

    def gen_loss(self):
        return get_loss(self.loss_name)()

    def gen_optimizer(self):
        return get_optimizer('ns_Optimizer')(self.model.parameters(), self.hyperparams)

    def train(self):
        LOSS = 0
        # max_jdg = 1e-8
        self.c += 1

        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            # eta = self.eta
            eta_alpha = self.eta
            X, y = self.get_next_train_batch()

            self.alpha_optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.s_model, _ = copy.deepcopy(self.alpha_optimizer.step(eta=eta_alpha))

            for i in range(self.prox_iters):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                bf = P2V(self.model.parameters())
                aft = P2V(self.optimizer.step(self.s_model))
                # if (bf - aft).norm() / bf.nelement() < max_jdg:
                #     break
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model)

            # u = P2V(self.s_model) - P2V(self.personalized_model)
            # lm = P2V(self.local_model)
            # lm_ = lm - self.learning_rate * self.lamda * u
            # V2P(lm_, self.local_model)
            for sm_param, pm_param, lm_param in zip(self.s_model, self.personalized_model, self.local_model):
                lm_param.data = lm_param.data - self.learning_rate * self.lamda * (sm_param.data - pm_param.data)
            self.set_model_parameters(self.local_model)

        return LOSS

    # def collect_grad(self, X, y):
    #     self.optimizer.zero_grad()
    #     output = self.model(X)
    #     loss = self.loss(output, y)
    #     loss.backward(retain_graph=True)
    #     for grad, model_p in zip(self.current_grad, self.model.parameters()):
    #         grad.data = model_p.grad.data.clone()
