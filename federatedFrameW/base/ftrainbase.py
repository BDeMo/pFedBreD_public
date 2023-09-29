from abc import abstractmethod, ABCMeta


class tbase(metaclass=ABCMeta):
    '''
    Base class for training.

    kwargs:
        - hyperparameters: hyperparameters for training

    abstract methods:
        - gen_locals: generate local calculations
        - gen_globals: generate global calculations
        - gen_nodes: generate nodes for federated training
        - gen_actions: generate actions for federated training
        - pre_train: pre-training
        - pre_action: pre-action
        - pre_local_train: pre-local training
        - train_local_model: train local model
        - pre_global_aggregate: pre-global aggregation
        - aggregate_global_model: aggregate global model
        - pre_local_eval: pre-local evaluation
        - eval_local_model: evaluate local model
        - pre_global_eval: pre-global evaluation
        - eval_global_model: evaluate global model
        - post_action: post-action
        - post_train: post-training

    '''

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.hyperparams = self.kwargs['hyperparams']
        if len(self.hyperparams) > 0:
            self.process_hyperparams()

        self.total_train_samples = 0  # total number of samples in the training set
        self.fnodes = []
        self.fglobals = []
        self.flocals = []
        self.actions = []
        self.graph = []

        self.gen_locals()
        self.gen_globals()
        self.gen_nodes()
        self.gen_actions()


    def train(self):
        self.pre_train()
        for action_fn in self.actions:
            self.pre_action(action_fn)
            action_fn()
            self.post_action(action_fn)
        self.post_train()

    # init
    def process_hyperparams(self):
        for hp in self.hyperparams.keys():
            setattr(self, hp, self.hyperparams[hp])

    @abstractmethod
    def gen_locals(self):
        pass

    @abstractmethod
    def gen_globals(self):
        pass

    @abstractmethod
    def gen_nodes(self):
        pass

    @abstractmethod
    def gen_actions(self):
        pass

    # train
    @abstractmethod
    def pre_train(self):
        pass

    @abstractmethod
    def pre_action(self, action):
        pass

    @abstractmethod
    def pre_local_train(self):
        pass

    @abstractmethod
    def train_local_model(self):
        pass

    @abstractmethod
    def pre_global_aggregate(self):
        pass

    @abstractmethod
    def aggregate_global_model(self):
        pass

    @abstractmethod
    def pre_local_eval(self):
        pass

    @abstractmethod
    def eval_local_model(self):
        pass

    @abstractmethod
    def pre_global_eval(self):
        pass

    @abstractmethod
    def eval_global_model(self):
        pass

    @abstractmethod
    def post_action(self, action):
        pass

    @abstractmethod
    def post_train(self):
        pass
