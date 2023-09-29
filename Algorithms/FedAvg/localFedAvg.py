from federatedFrameW.base.flocalbase import lbase
from federatedFrameW.utils.loss_utils import get_loss
from federatedFrameW.utils.optim_utils import get_optimizer


class lFedAvg(lbase):
    '''
    local Class for federated average

    kwargs:
        - id: local id
        - device: device
        - model: calculation model deepcopy
        - hyperparams:
            - batch_size: batch size
            - local_epochs: int, number of epochs for local training
            - optimizer_name: str, name of optimizer
            - loss: loss function
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gen_loss(self):
        return get_loss(self.loss_name)()

    def gen_optimizer(self):
        return get_optimizer(self.optimizer_name)(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
            self.clone_model_parameters(self.model.parameters(), self.local_model)
        return LOSS
