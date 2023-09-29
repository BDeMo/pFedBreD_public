import os
import re

import h5py
import numpy
import numpy as np
from matplotlib import pyplot as plt, cm
from numpy import mean
from scipy import interpolate

Centralized_Personalized_RegCoeff_eta_tau_methods = [
]

Centralized_Personalized_RegCoeff_eta_methods = ['pFedBreD'
                                                    , 'pFedBreD_ns'
                                                    , 'pFedBreD_ns_fm'
                                                    , 'pFedBreD_ns_fmd'
                                                    , 'pFedBreD_ns_lg'
                                                    , 'pFedBreD_ns_mh'
                                                    , 'pFedBreD_ns_meg_ft'
                                                    , 'pFedBreD_ns_lg_ft'
                                                    , 'pFedBreD_ns_mh_ft'
                                                    , 'pFedBreD_ns_meg'
                                                    , 'FedAMP'
                                                    , 'FedAMP_ft'
                                                    , 'pFedBayes'
                                                    , 'FedPAC'
                                                    , 'FedHN'
                                                    , 'Fedfomo'
                                                    , 'Ditto'
                                                 ] + Centralized_Personalized_RegCoeff_eta_tau_methods

Centralized_Personalized_RegCoeff_methods = [
                                                'pFedMe'
                                            ] + Centralized_Personalized_RegCoeff_eta_methods

Centralized_Personalized_methods = [
                                       'PerFedAvg'
                                       ,'PerFedAvg_ft'
                                       ,'FedEM'
                                       ,'FedEM_ft'
                                   ] + Centralized_Personalized_RegCoeff_methods
Centralized_methods = [
                          'FedAvg'
                      ] + Centralized_Personalized_methods

def get_all_files(file_dir):
    res = []
    for file in os.listdir(file_dir):
        res.append(file)
    return res


def get_dict_h5(file_names):
    res = dict()
    for file in file_names:
        k = re.findall(r'(.+?)_(\d)\.h5', file)
        if len(k) > 0:
            if k[0][0] not in res.keys():
                res[k[0][0]] = []
            res[k[0][0]].append(k[0][1])
    return res


def get_hparams_name(file_name):
    res = dict()
    k = re.findall(r'((((fe)|(fashion_))*(mnist))|(cifar10)|(sent140))'
                   r'_(((FedAvg)|(PerFedAvg)|pFedMe|pFedBreD_ns_lg|pFedBreD_ns_mh|pFedBreD_ns_meg|pFedBreD_kl_fo|pFedBreD_kl_mg|pFedBreD_kl_mfo|FedAMP|pFedBayes|FedEM|FedPAC|FedHN|Fedfomo|Ditto)(_ft)?)'
                   r'_((mclr)|(cnn)|(dnn)|(lstm))'
                   r'_(NLLLoss|CrossEntropyLoss|BCELoss|BCEWithLogitsLoss)'
                   r'(_(p|g))*'
                   r'_(.+?)lr'
                   r'_(.+?)nal'
                   r'_(.+?)bs'
                   r'_(.+?)b'
                   r'(_(.+?)le)*'
                   r'(_(.+?)pl)*'
                   r'(_(.+?)lam)*'
                   r'(_(.+?)pi)*'
                   r'(_(.+?)eta)*'
                   r'(.+?)*', file_name)
    if len(k) > 0:
        res = {
            'dataset': k[0][0]
            , 'name': k[0][8]
            , 'model_name': k[0][13]
            , 'loss_name': k[0][18]
            # general FL
            , 'tag': k[0][20]
            , 'learning_rate': float(k[0][21]) if len(k[0][21]) > 0 else ''
            , 'num_aggregate_locals': int(k[0][22]) if len(k[0][22]) > 0 else ''
            , 'batch_size': int(k[0][23]) if len(k[0][23]) > 0 else ''
            , 'beta': float(k[0][24]) if len(k[0][24]) > 0 else ''
            , 'local_epochs': int(k[0][26]) if len(k[0][26]) > 0 else ''
            # general PerFL
            , 'personal_learning_rate': float(k[0][28]) if len(k[0][28]) > 0 else ''
            # general PerFL with Reg or Bi-Level Optim
            , 'lamda': float(k[0][30]) if len(k[0][30]) > 0 else ''
            , 'prox_iters': int(k[0][32]) if len(k[0][32]) > 0 else ''
            , 'eta': float(k[0][34]) if len(k[0][34]) > 0 else ''
        }
    else:
        print('Unmatch')
    return res


def get_label_name(name, tag='g'):
    return name + '_' + tag


def simple_read_data(alg):
    print(alg)
    hf = h5py.File("./results/" + '{}.h5'.format(alg), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    rs_test_loss = np.array(hf.get('rs_test_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc, rs_test_loss


def average_smooth(data, window_len=20, window='hanning'):
    results = []
    if window_len < 3:
        return data
    for i in range(len(data)):
        x = data[i]
        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('numpy.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        results.append(y[window_len - 1:])
    return np.array(results)


def get_training_data_value(hyperparams):
    dataset = hyperparams['dataset']
    name = hyperparams['name']
    model_name = hyperparams['model_name']
    loss_name = hyperparams['loss_name']
    tag = hyperparams['tag']
    total_epochs = hyperparams['total_epochs']
    learning_rate = hyperparams['learning_rate']
    num_aggregate_locals = hyperparams['num_aggregate_locals']
    batch_size = hyperparams['batch_size']
    beta = hyperparams['beta']
    local_epochs = hyperparams['local_epochs']
    times = hyperparams['times']

    personal_learning_rate = hyperparams['personal_learning_rate']

    lamda = hyperparams['lamda']
    prox_iters = hyperparams['prox_iters']
    eta = hyperparams['eta']

    tau = hyperparams['tau']

    Numb_Algs = len(name)
    Numb_Glob_Iters = min(total_epochs)

    lr = hyperparams['learning_rate']

    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    test_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))

    for i in range(Numb_Algs):
        file_name = "[0-0]"
        if (name[i] in Centralized_methods):
            file_name = file_name + str(dataset[i]) \
                        + "_" + str(name[i]) \
                        + "_" + str(model_name[i]) \
                        + "_" + str(loss_name[i]) \
                        + "_" + str(tag[i]) \
                        + "_" + str(learning_rate[i]) + "lr" \
                        + "_" + str(num_aggregate_locals[i]) + "nal" \
                        + "_" + str(batch_size[i]) + "bs" \
                        + "_" + str(beta[i]) + "b" \
                        + "_" + str(local_epochs[i]) + "le"
        if (name[i] in Centralized_Personalized_methods):
            file_name = file_name + "_" + str(personal_learning_rate[i]) + "pl"
        if (name[i] in Centralized_Personalized_RegCoeff_methods):
            file_name = file_name + "_" + str(lamda[i]) + "lam" \
                        + "_" + str(prox_iters[i]) + "pi"
        if (name[i] in Centralized_Personalized_RegCoeff_eta_methods):
            file_name = file_name + "_" + str(eta[i]) + "eta"
        if (name[i] in Centralized_Personalized_RegCoeff_eta_tau_methods):
            file_name = file_name + "_" + str(tau[i]) + "tau"
        file_name = file_name + "_" + str(times[i])
        train_acc[i, :], train_loss[i, :], glob_acc[i, :], test_loss[i, :] = np.array(simple_read_data(file_name))[:,
                                                                             :Numb_Glob_Iters]

    return glob_acc, train_acc, train_loss, test_loss


def plot_acc(hyperparams, v='', labels=[], avg=False):
    plt.close()

    glob_acc_, train_acc_, train_loss_, test_loss_ = get_training_data_value(hyperparams)

    maxs = []
    bias = []
    for i in range(len(hyperparams['name'])):
        _max = glob_acc_[i].max()
        print("max accurancy:", hyperparams['name'][i], _max)
        maxs.append(_max)

    m_avg = mean(maxs)
    for _max in maxs:
        bias.append(abs(maxs - m_avg))
    print("max_bias:", format(m_avg * 100, ".2f") + '$\\pm$ ' + format(numpy.max(bias) * 100, ".3f"))
    glob_acc = average_smooth(glob_acc_, window='flat')
    # train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    # test_loss = average_smooth(test_loss_, window='flat')

    for acc, tag_data in [(glob_acc, 'Test'), (train_acc, 'Train')]:
        linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-.', '-.'] * 10
        markers = ["o", "v", "s", "*", "x", "P", '1', 'h', '<', '>', "p"] * 10
        colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm', '#f19790', '#f22790', '#f19220',
                  '#f22110',
                  '#f21710', '#f21110'] * 10
        plt.figure(1, figsize=(9, 9))
        plt.grid(True)
        for i in range(len(hyperparams['name'])):
            if len(labels) > 0:
                label = str(labels[i])
            else:
                label = get_label_name(hyperparams['name'][i], hyperparams['tag'][i])
            label = label  # + "-" + v + hyperparams['tag'][i]# + "-max:{0:.06}".format(str(glob_acc_[i].max()))
            plt.plot(acc[i, 1:], linestyle=linestyles[i], label=label, linewidth=1, color=colors[i], marker=markers[i],
                     markevery=0.2, markersize=5)
        if len(hyperparams['name']) > 12:
            fonts = 14
        else:
            fonts = 22
        plt.legend(loc='lower right', fontsize=fonts)
        plt.ylabel(tag_data + ' Accuracy', fontsize=fonts)
        plt.xlabel('Global rounds', fontsize=fonts)
        plt.title(tag_data + " Accuracy:" + v + hyperparams['dataset'][0] \
                  + '-' + hyperparams['tag'][0] + '-' + hyperparams['model_name'][0], fontsize=fonts)

        plot_max = acc.max()
        plot_min = plot_max
        for i in range(len(hyperparams['name'])):
            plot_min = min(acc[i][int(min(hyperparams['total_epochs']) / 2):].min(), plot_min)
        print([-0.3 * plot_max + 1.3 * plot_min, -0.05 * plot_min + 1.05 * plot_max])
        plt.ylim([-0.3 * plot_max + 1.3 * plot_min, -0.05 * plot_min + 1.05 * plot_max])
        print("./plot/" + hyperparams['dataset'][0] \
              + '-' + hyperparams['tag'][0] + '-' + hyperparams['model_name'][
                  0] + '-' + v + tag_data + "_acc.png")
        plt.savefig("./plot/" + hyperparams['dataset'][0] \
                    + '-' + hyperparams['tag'][0] + '-' + hyperparams['model_name'][
                        0] + '-' + v + tag_data + "_acc.png",
                    bbox_inches="tight")
        # plt.show()
        plt.close()


def plot_loss(hyperparams, v='', labels=[]):
    plt.close()

    glob_acc_, train_acc_, train_loss_, test_loss_ = get_training_data_value(hyperparams)

    for i in range(len(hyperparams['name'])):
        print("max accurancy:", hyperparams['name'][i], glob_acc_[i].max())

    # glob_acc = average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    # train_acc = average_smooth(train_acc_, window='flat')
    test_loss = average_smooth(test_loss_, window='flat')

    for loss, tag_data in [(test_loss, 'Test'), (train_loss, 'Train')]:
        linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-.', '-.'] * 10
        markers = ["o", "v", "s", "*", "x", "P", '1', 'h', '<', '>', "p"] * 10
        colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm', '#f19790', '#f22790', '#f19220',
                  '#f22110',
                  '#f21710', '#f21110'] * 10
        plt.figure(1, figsize=(9, 9))
        plt.grid(True)
        for i in range(len(hyperparams['name'])):
            if len(labels) > 0:
                label = str(labels[i])
            else:
                label = get_label_name(hyperparams['name'][i], hyperparams['tag'][i])
            label = label  # + "-" + v + hyperparams['tag'][i]# + "-max:{0:.06}".format(str(glob_acc_[i].max()))
            plt.plot(loss[i, 1:], linestyle=linestyles[i], label=label, linewidth=1, color=colors[i],
                     marker=markers[i],
                     markevery=0.2, markersize=5)

        if len(hyperparams['name']) > 12:
            fonts = 14
            # l_shift = 0.4
            # r_shift = 0.23
        else:
            fonts = 22
            # l_shift = 1
            # r_shift = 0.23
        # plt.legend(loc='lower right', bbox_to_anchor=(l_shift, r_shift), fontsize=fonts)
        plt.legend(loc='upper right', fontsize=fonts)
        plt.ylabel(tag_data + ' Loss', fontsize=fonts)
        plt.xlabel('Global rounds', fontsize=fonts)
        plt.title(tag_data + " Loss:" + v + hyperparams['dataset'][0] \
                  + '-' + hyperparams['tag'][0] + '-' + hyperparams['model_name'][0], fontsize=fonts)

        plot_min = loss.min()
        plot_max = plot_min
        for i in range(len(hyperparams['name'])):
            plot_max = max(loss[i][:int(max(hyperparams['total_epochs']) / 10)].max(), plot_max)
        print([1.05 * plot_min - 0.05 * plot_max, 0.85 * plot_max + 0.15 * plot_min])
        plt.ylim([1.05 * plot_min - 0.05 * plot_max, 0.85 * plot_max + 0.15 * plot_min])
        print("./plot/" + hyperparams['dataset'][0] \
              + '-' + hyperparams['tag'][0] + '-' + hyperparams['model_name'][
                  0] + '-' + v + tag_data + "_loss.png")
        plt.savefig("./plot/" + hyperparams['dataset'][0] \
                    + '-' + hyperparams['tag'][0] + '-' + hyperparams['model_name'][
                        0] + '-' + v + tag_data + "_loss.png",
                    bbox_inches="tight")
        plt.show()
        plt.close()


def plot_acc_3d_itpl(hyperparams, v='', labels=[]):
    plt.close()
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    glob_acc, train_acc, train_loss, test_loss = get_training_data_value(hyperparams)

    for i in range(len(hyperparams['name'])):
        print("max accurancy:", hyperparams['name'][i], glob_acc[i].max())

    plt.figure(1, figsize=(9, 9))
    plt.grid(True)

    # X = np.arange(0, len(labels), 1)
    X = np.array(labels)
    Y = np.arange(1, glob_acc.shape[1], 1)
    Z = glob_acc[:, 1:]
    _X = []
    _Y = []
    _Z = []
    for x in range(len(X)):
        for y in range(len(Y)):
            _X.append(X[x])
            _Y.append(Y[y])
            _Z.append(Z[x][y])
    smoothz = interpolate.interp2d(_X, _Y, _Z, kind='cubic')
    xNew = np.linspace(X.max(), X.min(), 1000)
    yNew = np.linspace(Y.max(), Y.min(), 1000)

    zNew = smoothz(xNew, yNew)
    xNew, yNew = np.meshgrid(xNew, yNew)
    # print(xNew.shape, yNew.shape, zNew.shape)
    surf = ax.plot_surface(xNew, yNew, zNew, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plot_max = glob_acc.max()
    plot_min = plot_max
    for i in range(len(hyperparams['name'])):
        plot_min = min(glob_acc[i][int(min(hyperparams['total_epochs']) / 100):].min(), plot_min)
    # print(plot_max, plot_min)
    ax.set_zlim(0, 1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(hyperparams['dataset'].upper() + "Non_Convex_Mnist_test_Com.pdf", bbox_inches="tight")
    plt.show()
    plt.close()


def plot_acc_3d_fit(hyperparams, v='', labels=[]):
    plt.close()
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    glob_acc, train_acc, train_loss, test_loss = get_training_data_value(hyperparams)

    for i in range(len(hyperparams['name'])):
        print("max accurancy:", hyperparams['name'][i], glob_acc[i].max())

    plt.figure(1, figsize=(9, 9))
    plt.grid(True)

    # X = np.arange(0, len(labels), 1)
    X = np.array(labels)
    Y = np.arange(1, glob_acc.shape[1], 1)
    Z = glob_acc[:, 1:].transpose()
    Xd, Yd = np.meshgrid(X, Y)
    surf = ax.plot_surface(Xd, Yd, Z, cmap='rainbow', linewidth=0, antialiased=False)

    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.title(" Test Accuracy: " + hyperparams['tag'][0] + '-' + v)

    plot_max = glob_acc.max()
    plot_min = plot_max
    for i in range(len(hyperparams['name'])):
        plot_min = min(glob_acc[i][int(min(hyperparams['total_epochs']) / 100):].min(), plot_min)
    # print(plot_max, plot_min)
    ax.set_zlim(-0.1 * plot_max + 1.1 * plot_min, -0.05 * plot_min + 1.05 * plot_max)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    print("./plot/" + hyperparams['dataset'][0] \
          + '-' + hyperparams['tag'][0] + '-' + hyperparams['model_name'][0] + v + "_3d.pdf")
    plt.savefig("./plot/" + hyperparams['dataset'][0] \
                + '-' + hyperparams['tag'][0] + '-' + hyperparams['model_name'][0] + v + "_3d.pdf", bbox_inches="tight")
    plt.show()
    plt.close()


def plot_loss_3d_itpl(hyperparams, v='', labels=[]):
    plt.close()
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    glob_acc, train_acc, train_loss, test_loss = get_training_data_value(hyperparams)

    for i in range(len(hyperparams['name'])):
        print("max accurancy:", hyperparams['name'][i], glob_acc[i].max())

    plt.figure(1, figsize=(9, 9))
    plt.grid(True)

    # X = np.arange(0, len(labels), 1)
    X = np.array(labels)
    Y = np.arange(1, train_loss.shape[1], 1)
    Z = train_loss[:, 1:]
    _X = []
    _Y = []
    _Z = []
    for x in range(len(X)):
        for y in range(len(Y)):
            _X.append(X[x])
            _Y.append(Y[y])
            _Z.append(Z[x][y])
    smoothz = interpolate.interp2d(_X, _Y, _Z, kind='cubic')
    xNew = np.linspace(X.max(), X.min(), 1000)
    yNew = np.linspace(Y.max(), Y.min(), 1000)

    zNew = smoothz(xNew, yNew)
    xNew, yNew = np.meshgrid(xNew, yNew)
    # print(xNew.shape, yNew.shape, zNew.shape)
    surf = ax.plot_surface(xNew, yNew, zNew, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plot_max = train_loss.max()
    plot_min = plot_max
    for i in range(len(hyperparams['name'])):
        plot_min = min(train_loss[i][int(min(hyperparams['total_epochs']) / 100):].min(), plot_min)
    # print(plot_max, plot_min)
    ax.set_zlim(0, 1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig(hyperparams['dataset'].upper() + "Non_Convex_Mnist_test_Com.pdf", bbox_inches="tight")
    plt.show()
    plt.close()
