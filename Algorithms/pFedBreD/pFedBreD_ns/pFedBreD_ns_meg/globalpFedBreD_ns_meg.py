from federatedFrameW.base.fglobalbase import gbase


class gpFedBreD_ns_meg(gbase):
    '''
    global Class for pFedBreD_ns_meg

    kwargs:
        - id: id of the global
        - model: calculation model deepcopy
        - hyperparame:
            - beta: global momentum
            - num_aggregate_locals: number of local models to aggregate
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_locals(self):
        self.sampled_locals = self.select_locals_uniform(num_aggregate_locals=self.num_aggregate_locals)
