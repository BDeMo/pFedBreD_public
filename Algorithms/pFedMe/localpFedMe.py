from federatedFrameW.base.flocalbase import lbase
from federatedFrameW.utils.loss_utils import get_loss
from federatedFrameW.utils.optim_utils import get_optimizer


class lpFedMe(lbase):
    '''
    local Class for pFedMe

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
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_optimizer = get_optimizer('SGD_lrs')(self.model.parameters(), self.hyperparams)

    def gen_loss(self):
        return get_loss(self.loss_name)()

    def gen_optimizer(self):
        return get_optimizer('ns_Optimizer')(self.model.parameters(), self.hyperparams)

    def train(self):
        LOSS = 0
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            X, y = self.get_next_train_batch()

            for i in range(self.prox_iters):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                p_m = self.optimizer.step(self.local_model)
            self.clone_model_parameters(p_m, self.personalized_model)

            for new_param, localweight in zip(self.personalized_model, self.local_model):
                if localweight.requires_grad:
                    localweight.data = localweight.data - self.lamda * self.learning_rate * (
                            localweight.data - new_param.data)

            self.set_model_parameters(self.local_model)
        return LOSS

