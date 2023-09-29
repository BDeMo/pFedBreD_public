import json
import os

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3


def general_CenFL_filename(obj, tag):
    return obj.dataset \
           + "_" + obj.name \
           + "_" + obj.model_name \
           + "_" + obj.loss_name \
           + tag \
           + "_" + str(obj.learning_rate) + "lr" \
           + "_" + str(obj.num_aggregate_locals) + "nal" \
           + "_" + str(obj.batch_size) + "bs" \
           + "_" + str(obj.beta) + "b" \
           + "_" + str(obj.local_epochs) + "le"


def general_CenPerFL_filename(obj, tag):
    return general_CenFL_filename(obj, tag) \
           + "_" + str(obj.personal_learning_rate) + "pl"


def general_CenPerFL_RegCoeff_filename(obj, tag):
    return general_CenPerFL_filename(obj, tag) \
           + "_" + str(obj.lamda) + "lam" \
           + "_" + str(obj.prox_iters) + "pi"


class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics_dir = os.path.join('out', self.params['dataset'], 'metrics_{}_{}_{}_{}_{}.json'.format(
            self.params['seed'], self.params['optimizer'], self.params['learning_rate'], self.params['num_epochs'],
            self.params['mu']))
        # os.mkdir(os.path.join('out', self.params['dataset']))
        if not os.path.exists('out'):
            os.mkdir('out')
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)
