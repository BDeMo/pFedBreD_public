from abc import abstractmethod

from federatedFrameW.fnode.nodeCentralizedFL import nCentralizedFL


class nCentralizedPerFL(nCentralizedFL):
    '''
    Centralized Personalized Federated Learning node.

    kwargs:
        - id: node id
        - fglobals: global calculations
        - flocals: local calculations
        - hyperparameters: hyperparameters

    implement methods:
        - sample_local: sample local calculations
        - aggregate_global: update global models
        - eval_global:evaluate global models
        - eval_local:evaluate local models
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval_local(self):
        res = []
        for fg in self.fglobals:
            res.append(fg.evaluate_personalized_model())
        return res
