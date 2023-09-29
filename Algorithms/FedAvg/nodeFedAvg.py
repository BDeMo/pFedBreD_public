from federatedFrameW.fnode.nodeCentralizedFL import nCentralizedFL


class nFedAvg(nCentralizedFL):
    '''
    Federated average node.

    kwargs:
        - id: node id
        - fglobals: global calculations
        - flocals: local calculations
        - hyperparameters: hyperparameters
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
