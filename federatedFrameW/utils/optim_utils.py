from torch.optim import Optimizer, SGD, Adam, Adagrad


def get_optimizer(optim_name):
    Optim_List = {
        'SGD': SGD
        , 'SGD_lrs': SGD_lrs
        , 'Adam': Adam
        , 'Adagrad': Adagrad
        , 'ns_Optimizer': ns_Optimizer
        , 'kl_Optimizer': kl_Optimizer
        , 'ns_iter_Optimizer': ns_iter_Optimizer
    }
    return Optim_List[optim_name]


class SGD_lrs(Optimizer):
    def __init__(self, params, hyperparams):
        lr = hyperparams['learning_rate']

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SGD_lrs, self).__init__(params, defaults)

    def step(self, eta=None):
        res_grad = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if eta:
                        p.data = p.data - eta * p.grad.data
                    else:
                        p.data = p.data - group['lr'] * p.grad.data
                    res_grad.append(p.grad.data)

        return group['params'], res_grad


class ns_iter_Optimizer(Optimizer):
    def __init__(self, params, hyperparams):
        lr = hyperparams['personal_learning_rate']
        lamda = hyperparams['lamda']

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr
                        , lamda=lamda
                        )
        super(ns_iter_Optimizer, self).__init__(params, defaults)

    def step(self, s_model):
        s_update = s_model.copy()
        for group in self.param_groups:
            for p, s_param in zip(group['params'], s_update):
                if p.requires_grad:
                    p.data = (p.data - group['lr'] * p.grad.data + group['lr'] * group['lamda'] * s_param.data) / (
                            1 + group['lr'] * group['lamda'])
        return group['params']


class ns_Optimizer(Optimizer):
    def __init__(self, params, hyperparams):
        lr = hyperparams['personal_learning_rate']
        lamda = hyperparams['lamda']

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr
                        , lamda=lamda
                        )
        super(ns_Optimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated):
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                if p.requires_grad:
                    p.data = p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - localweight.data)
                    )
        return group['params']


class kl_Optimizer(Optimizer):
    def __init__(self, params, hyperparams):
        lr = hyperparams['personal_learning_rate']
        lamda = hyperparams['lamda']

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr
                        , lamda=lamda
                        )
        super(kl_Optimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated):
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                if p.requires_grad:
                    p.data = p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - localweight.data)
                    )
        return group['params']
