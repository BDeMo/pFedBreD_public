from abc import abstractmethod

from federatedFrameW.base.fnodebase import nbase


class nCentralizedFL(nbase):
    '''
    The node class for centralized federated learning.

    kwargs:
        - id: node id
        - fglobals: global calculations
        - flocals: local calculations
        - hyperparams: hyperparameters

    implement methods:
        - sample_local: sample local calculations
        - aggregate_global: update global models
        - eval_global:evaluate global models
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connect()

    def connect(self):
        for fg in self.fglobals:
            fg.candidate_locals = self.flocals

    def train(self, sampled_only=False):
        if sampled_only:
            for fg in self.fglobals:
                fg.send_parameters_sampled_locals()
                for fl in fg.sampled_locals:
                    fl.train()
        else:
            for fg in self.fglobals:
                fg.send_parameters()
                for fl in fg.candidate_locals:
                    fl.train()

    def sample_local(self):
        for fg in self.fglobals:
            fg.sample_locals()

    def aggregate_global(self):
        for fg in self.fglobals:
            fg.aggregate_parameters(beta=self.beta)

    def eval_global(self):
        res = []
        for fg in self.fglobals:
            fg.send_parameters()
            res.append(fg.evaluate())
        return res
