from federatedFrameW.fnode.nodeCentralizedPerFL import nCentralizedPerFL


class nPerFedAvg(nCentralizedPerFL):
    '''
    PerFedAvg node.

    kwargs:
        - id: node id
        - fglobals: global calculations
        - flocals: local calculations
        - hyperparameters: hyperparameters
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gen_personalized_model(self):
        for fg in self.fglobals:
            fg.gen_personalized_model()