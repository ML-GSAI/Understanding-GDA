import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import adjusted_mutual_info_score
import futureproof
import sys
from models.gaussnb import GaussianNB

def save_vis_sim(m_step, errors, pic_path, args):
    plt.plot(m_step, np.mean(errors, axis=0), c='r', linewidth=1, label=args.model)
    plt.title(str(args.K) + ', ' + str(args.n))
    plt.xlabel('m')
    plt.ylabel('error')
    plt.legend()
    plt.savefig(pic_path)
    plt.close()

def save_vis(m_step, errors, pic_path, args):
    plt.plot(m_step, np.mean(errors, axis=0), c='r', linewidth=1, label=args.model)
    plt.title(args.backbone + ', ' + args.dataset)
    plt.xlabel('m')
    plt.ylabel('error')
    plt.legend()
    plt.savefig(pic_path)
    plt.close()

def show_deep_results(dataset, backbone, lr_path, nb_diag_path, nb_full_path, nb_low_rank_path, pic_dir, mode):
    if dataset == 'cifar10':
        if mode == 'long':
            m_step = [20,50,100,200,500,1000,2000,5000,10000,20000,30000,50000]
        else:
            m_step = [20,50,100,200,500,1000]
    else:
        K = 100
        if mode == 'long':
            m_step = [3*K,5*K,10*K,20*K,50*K,100*K,200*K,500*K]
        else:
            m_step = [3*K,5*K,10*K,20*K]

    error_lr = np.load(lr_path)
    print(error_lr.shape)
    error_nb_diag = np.load(nb_diag_path)
    if nb_full_path is not None:
        error_nb_full = np.load(nb_full_path)
    if nb_low_rank_path is not None:
        error_nb_low_rank = np.load(nb_low_rank_path)

    plt.figure()
    plt.plot(m_step, np.mean(error_lr[:,:len(m_step)], axis=0), marker='s', color='g', label='LR')
    plt.plot(m_step, np.mean(error_nb_diag[:,:len(m_step)], axis=0), marker='o', color='r', label='NB')
    if nb_full_path is not None:
        plt.plot(m_step, np.mean(error_nb_full[:,:len(m_step)], axis=0), linewidth=1, label='NB_full')
    if nb_low_rank_path is not None:
        plt.plot(m_step, np.mean(error_nb_low_rank[:,:len(m_step)], axis=0), linewidth=1, label='NB_diag_low rank')
    # plt.title(backbone + ', ' + dataset)
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.legend()
    plt.xlabel('m', labelpad=8, fontsize = 13)
    plt.ylabel('error', labelpad=8, fontsize = 13)
    plt.yticks(fontproperties = 'Times New Roman', size = 13)
    plt.xticks(fontproperties = 'Times New Roman', size = 13)
    pic_path = os.path.join(pic_dir, '%s.png' % (mode))
    plt.savefig(pic_path,bbox_inches='tight', dpi=800)
    plt.close()

def t_sne_results(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 't_sne', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    embedded_path_train = os.path.join(log_dir, 'embedded_train.npy')
    embedded_path_test = os.path.join(log_dir, 'embedded_test.npy')
    pic_path_train = os.path.join(log_dir, 't_sne_train.png')
    pic_path_test = os.path.join(log_dir, 't_sne_test.png')
    
    train_set_path = os.path.join(data_dir, 'train_features.npy')
    test_set_path = os.path.join(data_dir, 'val_features.npy')
    train_set = np.load(train_set_path)
    test_set = np.load(test_set_path)
    X_train, y_train = train_set[:,0:-1], train_set[:,-1]
    X_test, y_test = test_set[:,0:-1], test_set[:,-1]

    scaler = StandardScaler()
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
    
    X_train = scaler.fit_transform(X_train)
    X_train_embedded = tsne.fit_transform(X_train)
    np.save(embedded_path_train, X_train_embedded)
    plt.figure()
    plt.scatter(X_train_embedded[:,0], X_train_embedded[:,1], c=y_train, s=0.1, cmap='rainbow')
    plt.savefig(pic_path_train)
    plt.close()

    # X_test = scaler.fit_transform(X_test)
    # X_test_embedded = tsne.fit_transform(X_test)
    # np.save(embedded_path_test, X_test_embedded)
    # plt.figure()
    # plt.scatter(X_test_embedded[:,0], X_test_embedded[:,1], c=y_test, s=0.5, cmap='rainbow')
    # plt.savefig(pic_path_test)
    # plt.close()


def stats_features_means_vars(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'mean_vars', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    vars_path = os.path.join(log_dir, 'vars.npy')
    vars_pic_path = os.path.join(log_dir, 'vars.png')
    vars_p_path = os.path.join(log_dir, 'vars.csv')
    means_path = os.path.join(log_dir, 'means.npy')
    means_pic_path = os.path.join(log_dir, 'means.png')
    means_p_path = os.path.join(log_dir, 'means.csv')

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    train_features = train_features[:,0:-1]
    print(train_features.shape)
    vars = np.var(train_features, axis=0)
    means = np.mean(train_features, axis=0)
    np.save(vars_path, vars)
    np.save(means_path, means_path)

    vars_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        vars_p_value[0][p_idx] = p
        vars_p_value[1][p_idx] = np.percentile(vars, p)
    vars_p_value = pd.DataFrame(vars_p_value)
    vars_p_value.to_csv(vars_p_path, index=False)

    means_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        means_p_value[0][p_idx] = p
        means_p_value[1][p_idx] = np.percentile(means, p)
    means_p_value = pd.DataFrame(means_p_value)
    means_p_value.to_csv(means_p_path, index=False)

    plt.figure()
    plt.xlabel('var')
    plt.ylabel('counting')
    plt.hist(vars, bins=100)
    plt.savefig(vars_pic_path)
    plt.close()

    plt.figure()
    plt.xlabel('mean')
    plt.ylabel('counting')
    plt.hist(means, bins=100, density=True)
    plt.savefig(means_pic_path)
    plt.close()

def get_vars(samples:pd.DataFrame):
        samples = samples.iloc[:,:-1]
        vars = np.var(samples, axis=0) + 1e-9
        return vars

def stats_features_sigmas(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'sigmas_minmax', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    vars_path = os.path.join(log_dir, 'vars.npy')
    vars_pic_path = os.path.join(log_dir, 'vars.png')
    vars_p_path = os.path.join(log_dir, 'vars.csv')

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    print(train_features.shape)
    X_train, y_train = train_features[:,0:-1], train_features[:,-1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    train_features = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)

    train_features = pd.DataFrame(train_features)
    grouped = train_features.groupby([train_features.shape[1]-1])

    counts = grouped.agg('count').loc[:,0].values
    prior = counts / train_features.shape[0]
    vars_groupby = grouped.apply(get_vars).values
    vars = np.sum(vars_groupby * prior.reshape(-1,1), axis=0)

    np.save(vars_path, vars)

    vars_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        vars_p_value[0][p_idx] = p
        vars_p_value[1][p_idx] = np.percentile(vars, p)
    vars_p_value = pd.DataFrame(vars_p_value)
    vars_p_value.to_csv(vars_p_path, index=False)

    plt.figure()
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.xlabel(r'$\sigma_i^2$', labelpad=8, fontsize = 13)
    plt.ylabel('counting', labelpad=8, fontsize = 13)
    # 设置字体
    plt.yticks(fontproperties = 'Times New Roman', size = 13)
    plt.xticks(fontproperties = 'Times New Roman', size = 13)

    plt.hist(vars, bins=100)
    plt.savefig(vars_pic_path, bbox_inches='tight', dpi=800)
    plt.close()

from math import pi

def stats_features_gaussian_likelihood(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'likelihood_stand', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    likelihood_path = os.path.join(log_dir, 'likelihood.npy')
    likelihood_pic_path = os.path.join(log_dir, 'likelihood.png')
    likelihood_p_path = os.path.join(log_dir, 'likelihood.csv')

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    print(train_features.shape)
    X_train, y_train = train_features[:,0:-1], train_features[:,-1]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    train_features = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)

    train_features = pd.DataFrame(train_features)
    grouped = train_features.groupby([train_features.shape[1]-1])

    mus = grouped.agg('mean').values
    counts = grouped.agg('count').loc[:,0].values
    prior = counts / train_features.shape[0]
    vars_groupby = grouped.apply(get_vars).values
    vars = np.sum(vars_groupby * prior.reshape(-1,1), axis=0)

    # confidence actually
    likelihood = np.zeros((train_features.shape[0], train_features.shape[1]-1))
    train_features = train_features.values
    X_train, y_train = train_features[:,0:-1], train_features[:,-1]
    
    joint_log_likelihood = []
    # n_ij = 0
    K = len(set(list(y_train)))
    for i in range(K):
        n_ij = -0.5 * ((X_train - mus[i, :]) ** 2) / (vars.reshape(1,-1)) - 0.5 * np.log(2*pi*vars.reshape(1,-1))
        joint_log_likelihood.append(n_ij)
    joint_log_likelihood = np.array(joint_log_likelihood)
    joint_log_likelihood -= np.max(joint_log_likelihood, axis=0, keepdims=True)
    joint_log_likelihood = np.exp(joint_log_likelihood) / np.sum(np.exp(joint_log_likelihood), axis=0, keepdims=True)
    # print(joint_log_likelihood.min(), joint_log_likelihood.max())

    for i in range(joint_log_likelihood.shape[1]):
        for j in range(joint_log_likelihood.shape[2]):
            likelihood[i,j] = joint_log_likelihood[int(y_train[i]),i,j]
    # print(likelihood.max(), likelihood.min())

    likelihood = np.mean(likelihood, axis=0)
    np.save(likelihood_path, likelihood)
    likelihood_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        likelihood_p_value[0][p_idx] = p
        likelihood_p_value[1][p_idx] = np.percentile(likelihood, p)
    likelihood_p_value = pd.DataFrame(likelihood_p_value)
    likelihood_p_value.to_csv(likelihood_p_path, index=False)

    plt.figure()
    plt.xlabel('confidence')
    plt.ylabel('counting')
    plt.hist(likelihood, bins=100)
    plt.savefig(likelihood_pic_path)
    plt.close()

def get_kurtosis(samples:pd.DataFrame):
    samples = samples.iloc[:,:-1]
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples)
    X_4 = samples ** 4
    kurtosis = np.mean(X_4, axis=0) - 3
    kurtosis = np.abs(kurtosis)
    return kurtosis

def stats_features_kurtosis(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'kurtosis', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    kurtosis_path = os.path.join(log_dir, 'kurtosis.npy')
    kurtosis_pic_path = os.path.join(log_dir, 'kurtosis.png')
    kurtosis_p_path = os.path.join(log_dir, 'kurtosis.csv')

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    print(train_features.shape)
    train_features = pd.DataFrame(train_features)
    grouped = train_features.groupby([train_features.shape[1]-1])

    kurtosis_groupby = grouped.apply(get_kurtosis).values
    kurtosis = np.mean(kurtosis_groupby, axis=0)
    np.save(kurtosis_path, kurtosis)

    kurtosis_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        kurtosis_p_value[0][p_idx] = p
        kurtosis_p_value[1][p_idx] = np.percentile(kurtosis, p)
    kurtosis_p_value = pd.DataFrame(kurtosis_p_value)
    kurtosis_p_value.to_csv(kurtosis_p_path, index=False)

    plt.figure()
    plt.xlabel('kurtosis')
    plt.ylabel('counting')
    plt.hist(kurtosis, bins=100)
    plt.savefig(kurtosis_pic_path)
    plt.close()

def get_negentropy(samples:pd.DataFrame):
    samples = samples.iloc[:,:-1]
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples)

    X_3 = samples ** 3
    negentropy = np.mean(X_3, axis=0) ** 2

    X_4 = samples ** 4
    kurtosis = np.mean(X_4, axis=0) - 3

    negentropy = 1/12 * negentropy + 1/48 * (kurtosis ** 2)

    return negentropy

def stats_features_negentropy(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'negentropy', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    negentropy_path = os.path.join(log_dir, 'negentropy.npy')
    negentropy_pic_path = os.path.join(log_dir, 'negentropy.png')
    negentropy_p_path = os.path.join(log_dir, 'negentropy.csv')

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    print(train_features.shape)
    train_features = pd.DataFrame(train_features)
    grouped = train_features.groupby([train_features.shape[1]-1])

    negentropy_groupby = grouped.apply(get_negentropy).values
    negentropy = np.mean(negentropy_groupby, axis=0)
    np.save(negentropy_path, negentropy)

    negentropy_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        negentropy_p_value[0][p_idx] = p
        negentropy_p_value[1][p_idx] = np.percentile(negentropy, p)
    negentropy_p_value = pd.DataFrame(negentropy_p_value)
    negentropy_p_value.to_csv(negentropy_p_path, index=False)

    plt.figure()
    plt.xlabel('negentropy')
    plt.ylabel('counting')
    plt.hist(negentropy, bins=100)
    plt.savefig(negentropy_pic_path)
    plt.close()

def stats_features_KL(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'KL_gaussian', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    kl_diff_path = os.path.join(log_dir, 'kl_diff.npy')
    kl_pic_path = os.path.join(log_dir, 'kl.png')
    kl_p_path = os.path.join(log_dir, 'kl.csv')

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    print(train_features.shape)
    X_train, y_train = train_features[:,0:-1], train_features[:,-1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    joint_log_likelihood = nb.predict_gaussian_likehood(X_train)
    
    train_features = np.concatenate([joint_log_likelihood, y_train.reshape(-1,1)], axis=1)
    train_features = train_features[train_features[:,-1].argsort()]
    train_features_grouped = np.split(train_features[:,:-1], 10, axis=0)
    
    n = train_features.shape[1] - 1
    K = len(train_features_grouped)

    kl_diff = []
    for k in range(10):
        for k1 in range(10):
            for k2 in range(10):
                if k1 != k2:
                    kl_diff.append(np.mean(train_features_grouped[k][:,k1] - train_features_grouped[k][:,k2]))
    kl_diff = np.array(kl_diff)
    kl_diff = np.abs(kl_diff) / n
    np.save(kl_diff_path, kl_diff)

    kl_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        kl_p_value[0][p_idx] = p
        kl_p_value[1][p_idx] = np.percentile(kl_diff, p)
    kl_p_value = pd.DataFrame(kl_p_value)
    kl_p_value.to_csv(kl_p_path, index=False)

    plt.figure()
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.xlabel(r'$\vert\beta_{k_1,k_2,k}\vert$', labelpad=8, fontsize = 13)
    plt.ylabel('counting', labelpad=8, fontsize = 13)
    # 设置字体
    plt.yticks(fontproperties = 'Times New Roman', size = 13)
    plt.xticks(fontproperties = 'Times New Roman', size = 13)
    plt.xlabel(r'$\vert\beta_{k_1,k_2,k}\vert$')
    plt.ylabel('counting')

    plt.hist(kl_diff, bins=100)
    plt.savefig(kl_pic_path, bbox_inches='tight', dpi=800)
    plt.close()

# def stats_features_KL(dataset, backbone):
#     data_dir = os.path.join('./datasets', backbone, dataset)
#     log_dir = os.path.join('./stats', 'KL', backbone, dataset)
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir, exist_ok=True)
#     kl_path = os.path.join(log_dir, 'kl.npy')
#     kl_diff_path = os.path.join(log_dir, 'kl_diff.npy')
#     kl_pic_path = os.path.join(log_dir, 'kl.png')
#     kl_p_path = os.path.join(log_dir, 'kl.csv')
#     bins = 1000

#     train_set_path = os.path.join(data_dir, 'train_features.npy')
#     train_features = np.load(train_set_path)
#     print(train_features.shape)
#     X_train, y_train = train_features[:,0:-1], train_features[:,-1]
#     scaler = MinMaxScaler()
#     X_train = scaler.fit_transform(X_train)
#     train_features = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)

#     train_features = train_features[train_features[:,-1].argsort()]
#     train_features = (train_features * bins).astype(int)
#     # print(train_features.min(), train_features.max())

#     train_features_grouped = np.split(train_features[:,:-1], 10, axis=0)
#     n = train_features.shape[1] - 1
#     hist_grouped = 1e-9 * np.ones((len(train_features_grouped), n, bins))
#     for i in tqdm(range(hist_grouped.shape[0])):
#         for j in tqdm(range(hist_grouped.shape[1])):
#             hist_grouped[i,j,:] = np.histogram(train_features_grouped[i][:,j],bins=bins,density=False)[0]
#     hist_grouped += 1e-5
#     hist_grouped /= np.sum(hist_grouped, axis=2).reshape((hist_grouped.shape[0],hist_grouped.shape[1],1))
#     # print(np.sum(hist_grouped[0], axis=1))
    
#     kl = np.zeros((10,10))
#     for i in range(kl.shape[0]):
#         for j in range(kl.shape[1]):
#             if i != j:
#                 P = hist_grouped[i]
#                 Q = hist_grouped[j]
#                 kl_pointwise = P * (np.log(P) - np.log(Q))
#                 kl[i,j] = np.sum(kl_pointwise)
#     np.save(kl_path, kl)

#     kl_diff = []
#     for k in range(kl.shape[0]):
#         for k1 in range(kl.shape[0]):
#             for k2 in range(kl.shape[0]):
#                 if k1 != k2:
#                     kl_diff.append(kl[k][k1] - kl[k][k2])
#     kl_diff = np.array(kl_diff)
#     kl_diff = np.abs(kl_diff) / n
#     np.save(kl_diff_path, kl_diff)

#     kl_p_value = np.zeros((2, 11))
#     for p_idx, p in enumerate(range(0,101,10)):
#         kl_p_value[0][p_idx] = p
#         kl_p_value[1][p_idx] = np.percentile(kl_diff, p)
#     kl_p_value = pd.DataFrame(kl_p_value)
#     kl_p_value.to_csv(kl_p_path, index=False)

#     plt.figure()
#     plt.xlabel('beta')
#     plt.ylabel('counting')
#     plt.hist(kl_diff, bins=100)
#     plt.savefig(kl_pic_path)
#     plt.close()

def stats_features_var_likelihood_diff(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'var_likelihood_diff', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    kl_diff_path = os.path.join(log_dir, 'var_likelihood_diff.npy')
    kl_pic_path = os.path.join(log_dir, 'var_likelihood_diff.png')
    kl_p_path = os.path.join(log_dir, 'var_likelihood_diff.csv')

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    print(train_features.shape)
    X_train, y_train = train_features[:,0:-1], train_features[:,-1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    joint_log_likelihood = nb.predict_gaussian_likehood(X_train)
    
    train_features = np.concatenate([joint_log_likelihood, y_train.reshape(-1,1)], axis=1)
    train_features = train_features[train_features[:,-1].argsort()]
    train_features_grouped = np.split(train_features[:,:-1], 10, axis=0)
    
    n = train_features.shape[1] - 1
    K = len(train_features_grouped)

    kl_diff = []
    for k in range(10):
        for k1 in range(10):
            for k2 in range(10):
                if k1 != k2:
                    kl_diff.append(np.var(train_features_grouped[k][:,k1] - train_features_grouped[k][:,k2]))
    kl_diff = np.array(kl_diff)
    kl_diff = np.abs(kl_diff) / n
    np.save(kl_diff_path, kl_diff)

    kl_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        kl_p_value[0][p_idx] = p
        kl_p_value[1][p_idx] = np.percentile(kl_diff, p)
    kl_p_value = pd.DataFrame(kl_p_value)
    kl_p_value.to_csv(kl_p_path, index=False)

    plt.figure()
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.xlabel(r'$\vert\alpha_{k_1,k_2,k}\vert$', labelpad=8, fontsize = 13)
    plt.ylabel('counting', labelpad=8, fontsize = 13)
    # 设置字体
    plt.yticks(fontproperties = 'Times New Roman', size = 13)
    plt.xticks(fontproperties = 'Times New Roman', size = 13)

    plt.hist(kl_diff, bins=100)
    plt.savefig(kl_pic_path, bbox_inches='tight', dpi=800)
    plt.close()

def adjusted_mutual_info(i, j):
    ans = adjusted_mutual_info_score(sample[:,i], sample[:,j])
    return ans

def get_mutual_info(samples:pd.DataFrame):
    global sample
    sample = samples.iloc[:,:-1].values
    pairs = []
    n = sample.shape[1]
    for i in range(n-1):
        for j in range(i+1, n):
            pairs.append((i,j))
    #with ProcessPoolExecutor() as pool:
        # mutual_info = list(pool.map(adjusted_mutual_info, pairs))
    # mutual_info = list(map(adjusted_mutual_info, pairs))
    mutual_info = []
    ex = futureproof.ProcessPoolExecutor()
    with futureproof.TaskManager(
        ex, error_policy=futureproof.ErrorPolicyEnum.RAISE
    ) as tm:
        tm.map(adjusted_mutual_info, pairs)
        for task in tm.as_completed():
            if isinstance(task.result, Exception):
                print("%r generated an exception: %s" % (task.args[0], task.result))
            else:
                mutual_info.append(task.result)
    mutual_info = pd.Series(mutual_info)
    return mutual_info

def stats_features_mutual_info(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'MI', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    mi_path = os.path.join(log_dir, 'mi.npy')
    mi_pic_path = os.path.join(log_dir, 'mi.png')
    mi_p_path = os.path.join(log_dir, 'mi.csv')
    bins = 1000

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    print(train_features.shape)
    X_train, y_train = train_features[:,0:-1], train_features[:,-1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = (X_train * bins)
    train_features = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1).astype(int)
    train_features = pd.DataFrame(train_features)

    grouped = train_features.groupby([train_features.shape[1]-1])
    mi_groupby = grouped.apply(get_mutual_info)
    print(mi_groupby.shape)
    np.save(mi_path, mi)
    mi = np.mean(mi_groupby, axis=1)
    # np.save(mi_path, mi)

    mi_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        mi_p_value[0][p_idx] = p
        mi_p_value[1][p_idx] = np.percentile(mi, p)
    mi_p_value = pd.DataFrame(mi_p_value)
    mi_p_value.to_csv(mi_p_path, index=False)

    plt.figure()
    plt.xlabel('mutual information')
    plt.ylabel('counting')
    plt.hist(mi, bins=10)
    plt.savefig(mi_pic_path)
    plt.close()

def stats_features_means_vars_group(features_dir, pic_dir):
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    train_set_path = os.path.join(features_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    tab = pd.DataFrame(train_features)
    grouped = tab.groupby(tab.shape[1]-1)
    means_by_group = grouped.agg('mean')
    vars_by_group = (grouped.agg('var'))

    for i in range(10):
        plt.figure()
        plt.title('Cifar10 Means')
        plt.xlabel('mean')
        plt.ylabel('counting')
        plt.hist(means_by_group.iloc[i], bins=100)
        plt.savefig(os.path.join(pic_dir, 'cifar10_means_K{}.png'.format(i)))
        plt.close()

        plt.figure()
        plt.title('Cifar10 Vars')
        plt.xlabel('var')
        plt.ylabel('counting')
        plt.hist(vars_by_group.iloc[i], bins=100)
        plt.savefig(os.path.join(pic_dir, 'cifar10_vars_K{}.png'.format(i)))
        plt.close()
    
def stats_features_cov_group(features_dir, pic_dir):
    if not os.path.exists(pic_dir):
        os.mkdir(pic_dir)
    train_set_path = os.path.join(features_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    tab = pd.DataFrame(train_features)
    grouped = tab.groupby(tab.shape[1]-1)

    for i, values in (grouped):
        # print(i)
        # print(values.to_numpy()[:,0:-1].shape)
        values = values.to_numpy()[:,0:-1]
        values = values.T + 1e-9
        coef = np.abs(np.corrcoef(values))
        # print(coef)
        print('--------------------')
        print(i)
        print((np.sum(coef) - coef.shape[0]) / (coef.shape[0] ** 2 - coef.shape[0]))
        print(np.sum(coef >= 0.3) / 2, (coef.shape[0] ** 2 - coef.shape[0]) / 2, np.sum(coef >= 0.3)  / (coef.shape[0] ** 2 - coef.shape[0]))
        print(np.sum(coef >= 0.5) / 2, (coef.shape[0] ** 2 - coef.shape[0]) / 2, np.sum(coef >= 0.5)  / (coef.shape[0] ** 2 - coef.shape[0]))
        print(np.sum(coef >= 0.8) / 2, (coef.shape[0] ** 2 - coef.shape[0]) / 2, np.sum(coef >= 0.8)  / (coef.shape[0] ** 2 - coef.shape[0]))
        print('--------------------')

if __name__ == '__main__':
    pass