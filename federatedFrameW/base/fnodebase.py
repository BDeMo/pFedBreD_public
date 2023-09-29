from abc import abstractmethod, ABCMeta


class nbase(metaclass=ABCMeta):
    '''
    Abstract class for federated node

    kwargs:
        - id: node id
        - fglobals: global calculations
        - flocals: local calculations
        - hyperparams: hyperparameters

    abstract methods:
        - train: train the models
        - connect: connect globals and locals
    '''

    def __init__(self, *args, **kwargs):
        self.id = kwargs['id']
        self.fglobals = kwargs['fglobals']
        self.flocals = kwargs['flocals']
        self.hyperparams = kwargs['hyperparams']
        if len(self.hyperparams) > 0:
            self.process_hyperparams()

        self.total_train_samples = 0

    def process_hyperparams(self):
        for hp in self.hyperparams.keys():
            setattr(self, hp, self.hyperparams[hp])

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def train(self):
        pass
