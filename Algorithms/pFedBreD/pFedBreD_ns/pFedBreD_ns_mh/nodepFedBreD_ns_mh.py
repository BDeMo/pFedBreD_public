from federatedFrameW.fnode.nodeCentralizedPerFL import nCentralizedPerFL


class npFedBreD_ns_mh(nCentralizedPerFL):
    '''
    pFedBreD_ns_mh node.

    kwargs:
        - id: node id
        - fglobals: global calculations
        - flocals: local calculations
        - hyperparameters: hyperparameters
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
