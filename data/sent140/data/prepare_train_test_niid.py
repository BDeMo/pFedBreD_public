from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os

from federatedFrameW.utils.language_utils import line_to_indices
from federatedFrameW.utils.language_utils import get_word_emb_arr

random.seed(1)
np.random.seed(1)
NUM_USERS = 10
NUM_MIN_SAMPLES = 10
# TRAIN_TEST_FRACTION = 0.1
ALPHA = 0.5

train_path = './train/all_data_niid_05_keep_3_train_9.json'
train_output_path = './train/train.json'
test_path = './test/all_data_niid_05_keep_3_test_9.json'
test_output_path = './test/test.json'

VOC_PATH = '../../../federatedFrameW/utils/glove/embs.json'
embs_output_path = './embs/embs.json'


def gen_parts(n_total, NUM_USERS, NUM_MIN_SAMPLES):
    n_dir = n_total - NUM_USERS * NUM_MIN_SAMPLES
    partitions = np.array([NUM_MIN_SAMPLES] * NUM_USERS, dtype=np.int64)
    d_partitions = partitions + np.array(n_dir * np.random.dirichlet(alpha=ALPHA * np.ones(NUM_USERS)),
                                         dtype=np.int64)
    n_res = n_total - d_partitions.sum()
    for i in range(n_res):
        index = random.randint(0, NUM_USERS - 1)
        d_partitions[index] += 1

    return d_partitions


def seqs_2_index(seqs, indd):
    res = []
    for seq in seqs:
        res.append(line_to_indices(seq.lower(), indd))

    return res


def compress_by_rerank(train_data, test_data, word_emb_arr, indd, vocab):
    _train_data = {'users': train_data['users'], 'user_data': {}, 'num_samples': train_data['num_samples']}
    _test_data = {'users': test_data['users'], 'user_data': {}, 'num_samples': test_data['num_samples']}
    embedings = []
    rank_dict = {}
    rank = 0

    print(test_data['user_data'].keys())
    for i, uname in enumerate(train_data['users'], 0):
        for _d_train_x in train_data['user_data'][uname]['x']:
            for w_train_x in _d_train_x:
                if w_train_x in rank_dict.keys():
                    rank_dict[w_train_x] += 1
                else:
                    rank_dict[w_train_x] = 1
        for _d_test_x in test_data['user_data'][uname]['x']:
            for w_test_x in _d_test_x:
                if w_test_x in rank_dict.keys():
                    rank_dict[w_test_x] += 1
                else:
                    rank_dict[w_test_x] = 1

    o2n = {}
    n2o = {}
    st = sorted(rank_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for k, (v, c) in enumerate(st, 0):
        n2o[k] = v
        o2n[v] = k
        if v >= len(word_emb_arr):
            embedings.append([0.] * word_emb_arr[0].shape[0])
        else:
            embedings.append(word_emb_arr[v].tolist())

    for i in st[0:4]:
        if i[0] < len(vocab):
            print(vocab[i[0]], ':', o2n[i[0]], i[0])
        else:
            print('PAD:', o2n[i[0]], i[0])
        print(embedings[o2n[i[0]]])

    for uname in train_data['user_data'].keys():
        res_train_x = []
        res_test_x = []
        for user_train_data in train_data['user_data'][uname]['x']:
            _user_train_data = []
            for e_user_train_data in user_train_data:
                _user_train_data.append(o2n[e_user_train_data])
            res_train_x.append(_user_train_data)

        for user_test_data in test_data['user_data'][uname]['x']:
            _user_test_data = []
            for e_user_test_data in user_test_data:
                _user_test_data.append(o2n[e_user_test_data])
            res_test_x.append(_user_test_data)

        _train_data['user_data'][uname] = {
            'x': res_train_x
            , 'y': train_data['user_data'][uname]['y']
        }
        _test_data['user_data'][uname] = {
            'x': res_test_x
            , 'y': test_data['user_data'][uname]['y']
        }

    return _train_data, _test_data, embedings, len(st)


if __name__ == '__main__':
    with open(train_path, 'r') as fp:
        train_raw_data = json.load(fp)
    with open(test_path, 'r') as fp:
        test_raw_data = json.load(fp)

    word_emb_arr, indd, vocab = get_word_emb_arr(VOC_PATH)

    n_total = len(train_raw_data['users'])
    parts = gen_parts(n_total, NUM_USERS, NUM_MIN_SAMPLES)
    print(parts)

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    setoff = 0
    for i in trange(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        train_data['users'].append(uname)
        test_data['users'].append(uname)
        _d_train_x, _d_train_y, _d_test_x, _d_test_y = [], [], [], []
        for d_train, d_test in zip([i[1] for i in train_raw_data['user_data'].items()][setoff:setoff + parts[i]],
                                   [i[1] for i in test_raw_data['user_data'].items()][setoff:setoff + parts[i]]):
            _d_train_x += [d[4] for d in d_train['x']]
            _d_train_y += d_train['y']
            _d_test_x += [d[4] for d in d_test['x']]
            _d_test_y += d_test['y']
        setoff += parts[i]

        _d_train_x = seqs_2_index(_d_train_x, indd)
        _d_test_x = seqs_2_index(_d_test_x, indd)

        train_data['user_data'][uname] = {
            'x': _d_train_x
            , 'y': _d_train_y
        }
        train_data['num_samples'].append(len(_d_train_x))

        test_data['user_data'][uname] = {
            'x': _d_test_x
            , 'y': _d_test_y
        }
        test_data['num_samples'].append(len(_d_test_x))

    # r_d, r_l = [], []
    # for i in data['user_data'].items():
    #     datas, labels = i[1]['x'], i[1]['y']
    #     for d, l in zip(datas, labels):
    #         r_d.append(d[4])
    #         r_l.append(l)
    # for d, l in zip(r_d, r_l):
    #     print(d, l)
    train_data, test_data, embedings, num_rank = compress_by_rerank(train_data, test_data, word_emb_arr, indd, vocab)
    print(num_rank)

    print('-' * 40 + 'dumping train data' + '-' * 40)
    with open(train_output_path, 'w') as outfile:
        print(train_output_path)
        json.dump(train_data, outfile)
    print('-' * 40 + 'dumping test data' + '-' * 40)
    with open(test_output_path, 'w') as outfile:
        print(test_output_path)
        json.dump(test_data, outfile)
    print('-' * 40 + 'dumping embedings' + '-' * 40)
    with open(embs_output_path, 'w') as outfile:
        print(embs_output_path)
        json.dump(embedings, outfile)

    print("Finish Generating Samples")
