import argparse
from sklearn.datasets import make_blobs
from models.gaussnb import GaussianNB
import numpy as np
from utils.tools import *

import sys

parser = argparse.ArgumentParser(description='Simulation experiments')

parser.add_argument('--train_size_list', default=[40], type=list, nargs='+',
                    help='train size')
parser.add_argument('--gamma_list', default=[0, 1, 2, 5, 10, 20, 50, 100], type=list, nargs='+',
                    help='control the generated samples')
parser.add_argument('--d_list', default=[1], type=list, nargs='+',
                    help='feature dimension')
parser.add_argument('--sigma', default=0.6, type=float,
                    help='sigma of mixture Gaussian.')

parser.add_argument('--mode', default='gamma', type=str, choices=['d', 'm', 'gamma'],
                    help='easy or hard.')
parser.add_argument('--repeat', default=1000, type=int, metavar='N',
                    help='repeat times')

def get_mus_d(d):
    mus = [d*[-1/np.sqrt(d)], d*[1/np.sqrt(d)]]
    return mus

def get_zero_one_loss(theta, X, y):
    pred = np.sum(theta.reshape(1,-1) * X, axis=1)
    pred[pred >= 0] = 1
    pred[pred < 0] = 0
    return 1 - np.sum(pred == y) / len(y)

def get_NLL_loss(theta, X, y:np.array, d, sigma):
    Theta = np.repeat(theta.reshape(1,-1), len(y), axis=0)
    y = 2 * y - 1
    nll = np.sum((X - y.reshape(-1,1) * Theta) ** 2)
    nll = nll / (2 * (sigma ** 2) * len(y))
    return nll

def main():
    args = parser.parse_args()

    if args.mode == 'gamma':
        for d in args.d_list:
            for m in args.train_size_list:
                args.d = d
                args.m = m
                log_dir = os.path.join('./log/bGMM/gamma', ('d%d_m%d'% (args.d, args.m)), ('sigma%.2f'% args.sigma))
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                logger = get_console_file_logger(name='%s, %s' % (args.mode, ('d%d_m%d'% (args.d, args.m))), logdir=log_dir)
                logger.info(args._get_kwargs())
                simulation_gamma(args, log_dir, logger)

    elif args.mode == 'm': # x-axis is m
        for d in args.d_list:
            args.d = d
            log_dir = os.path.join('./log/bGMM/m', ('d%d'% (args.d)), ('sigma%.2f'% args.sigma))
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            logger = get_console_file_logger(name='%s, %s' % (args.mode, ('d%d'% (args.d))), logdir=log_dir)
            logger.info(args._get_kwargs())
            simulation_m(args, log_dir, logger)

    elif args.mode == 'd': #x-axis is d
        for m in args.train_size_list:
            args.m = m
            log_dir = os.path.join('./log/bGMM/d', ('m%d'% (args.m)), ('sigma%.2f'% args.sigma))
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            logger = get_console_file_logger(name='%s, %s' % (args.mode, ('m%d'% (args.m))), logdir=log_dir)
            logger.info(args._get_kwargs())
            simulation_d(args, log_dir, logger)

def simulation_gamma(args, log_dir, logger):
    zero_one_loss_path_2 = os.path.join(log_dir, 'zero_one_loss_2.npy')
    nll_loss_path_2 = os.path.join(log_dir, 'nll_loss_2.npy')

    zero_one_loss_2 = np.zeros((len(args.gamma_list), args.repeat))
    nll_loss_2 = np.zeros((len(args.gamma_list), args.repeat))

    for t in range(args.repeat):

        mus = get_mus_d(args.d)
        X_train, y_train = make_blobs(n_samples=args.m, n_features=args.d, centers=mus,
                            cluster_std=[args.sigma, args.sigma])
        X_test, y_test = make_blobs(n_samples=10000, n_features=args.d, centers=mus,
                            cluster_std=[args.sigma, args.sigma])

        nb = GaussianNB()
        nb.fit(X_train, y_train)
        vars_nb = np.sum(nb.vars_groupby * nb.prior.reshape(-1,1), axis=0).reshape(1,-1)

                        
        for j, gamma in enumerate(args.gamma_list):

            aug_size = int(gamma * args.m)
            if aug_size > 0:
                X_new, y_new = make_blobs(aug_size, n_features=args.d, centers=[nb.mus[0,:],nb.mus[1,:]],
                                            cluster_std=[np.sqrt(vars_nb), np.sqrt(vars_nb)])
                X_aug = np.concatenate([X_train, X_new], axis=0)
                y_aug = np.concatenate([y_train, y_new])
            else:
                X_aug = X_train
                y_aug = y_train

            theta_aug = np.sum(X_aug * (2 * y_aug-1).reshape(-1,1), axis=0) / len(y_aug)
            zero_one_loss_theta2_train = get_zero_one_loss(theta_aug, X_aug, y_aug)
            nll_loss_theta2_train = get_NLL_loss(theta_aug, X_aug, y_aug, args.d, args.sigma)
            zero_one_loss_theta2_test = get_zero_one_loss(theta_aug, X_test, y_test)
            nll_loss_theta2_test = get_NLL_loss(theta_aug, X_test, y_test, args.d, args.sigma)

            zero_one_loss_theta2_gap = np.abs(zero_one_loss_theta2_train - zero_one_loss_theta2_test)
            nll_loss_theta2_gap = np.abs(nll_loss_theta2_train - nll_loss_theta2_test)

            logger.info('t = %d, train size = %d, gamma = %.2f, aug_size = %d, 01 loss train = %.4f, 01 loss test = %.4f, gap = %.4f' % \
                        (t+1, args.m, gamma, aug_size, zero_one_loss_theta2_train, zero_one_loss_theta2_test, zero_one_loss_theta2_gap))
            logger.info('t = %d, train size = %d, gamma = %.2f, aug_size = %d, nll loss train = %.4f, nll loss test = %.4f, gap = %.4f' % \
                        (t+1, args.m, gamma, aug_size, nll_loss_theta2_train, nll_loss_theta2_test, nll_loss_theta2_gap))
                
            zero_one_loss_2[j,t] = zero_one_loss_theta2_gap
            nll_loss_2[j,t] = nll_loss_theta2_gap
            
    np.save(zero_one_loss_path_2, zero_one_loss_2)
    np.save(nll_loss_path_2, nll_loss_2)



def simulation_m(args, log_dir, logger):
    zero_one_loss_path_2 = os.path.join(log_dir, 'zero_one_loss_2.npy')
    nll_loss_path_2 = os.path.join(log_dir, 'nll_loss_2.npy')

    zero_one_loss_2 = np.zeros((len(args.train_size_list), len(args.gamma_list), args.repeat))
    nll_loss_2 = np.zeros((len(args.train_size_list), len(args.gamma_list), args.repeat))

    for i, train_size in enumerate(args.train_size_list):

        for t in range(args.repeat):

            mus = get_mus_d(args.d)
            X_train, y_train = make_blobs(n_samples=train_size, n_features=args.d, centers=mus,
                                cluster_std=[args.sigma, args.sigma])
            X_test, y_test = make_blobs(n_samples=10000, n_features=args.d, centers=mus,
                                cluster_std=[args.sigma, args.sigma])


            nb = GaussianNB()
            nb.fit(X_train, y_train)
            vars_nb = np.sum(nb.vars_groupby * nb.prior.reshape(-1,1), axis=0).reshape(1,-1)
                        
            for j, gamma in enumerate(args.gamma_list):

                aug_size = int(gamma * train_size)
                if aug_size > 0:
                    X_new, y_new = make_blobs(aug_size, n_features=args.d, centers=[nb.mus[0,:],nb.mus[1,:]],
                                                cluster_std=[np.sqrt(vars_nb), np.sqrt(vars_nb)])
                    X_aug = np.concatenate([X_train, X_new], axis=0)
                    y_aug = np.concatenate([y_train, y_new])
                else:
                    X_aug = X_train
                    y_aug = y_train

                theta_aug = np.sum(X_aug * (2 * y_aug-1).reshape(-1,1), axis=0) / len(y_aug)
                zero_one_loss_theta2_train = get_zero_one_loss(theta_aug, X_aug, y_aug)
                nll_loss_theta2_train = get_NLL_loss(theta_aug, X_aug, y_aug, args.d, args.sigma)
                zero_one_loss_theta2_test = get_zero_one_loss(theta_aug, X_test, y_test)
                nll_loss_theta2_test = get_NLL_loss(theta_aug, X_test, y_test, args.d, args.sigma)

                zero_one_loss_theta2_gap = np.abs(zero_one_loss_theta2_train - zero_one_loss_theta2_test)
                nll_loss_theta2_gap = np.abs(nll_loss_theta2_train - nll_loss_theta2_test)

                logger.info('t = %d, train size = %d, gamma = %.2f, aug_size = %d, 01 loss train = %.4f, 01 loss test = %.4f, gap = %.4f' % \
                            (t+1, train_size, gamma, aug_size, zero_one_loss_theta2_train, zero_one_loss_theta2_test, zero_one_loss_theta2_gap))
                logger.info('t = %d, train size = %d, gamma = %.2f, aug_size = %d, nll loss train = %.4f, nll loss test = %.4f, gap = %.4f' % \
                            (t+1, train_size, gamma, aug_size, nll_loss_theta2_train, nll_loss_theta2_test, nll_loss_theta2_gap))
                
                zero_one_loss_2[i,j,t] = zero_one_loss_theta2_gap
                nll_loss_2[i,j,t] = nll_loss_theta2_gap
            
    np.save(zero_one_loss_path_2, zero_one_loss_2)
    np.save(nll_loss_path_2, nll_loss_2)


def simulation_d(args, log_dir, logger):
    zero_one_loss_path_2 = os.path.join(log_dir, 'zero_one_loss_2.npy')
    nll_loss_path_2 = os.path.join(log_dir, 'nll_loss_2.npy')

    zero_one_loss_2 = np.zeros((len(args.d_list), len(args.gamma_list), args.repeat))
    nll_loss_2 = np.zeros((len(args.d_list), len(args.gamma_list), args.repeat))

    for i, d in enumerate(args.d_list):

        for t in range(args.repeat):

            mus = get_mus_d(d)
            X_train, y_train = make_blobs(n_samples=args.m, n_features=d, centers=mus,
                                cluster_std=[args.sigma, args.sigma])
            X_test, y_test = make_blobs(n_samples=10000, n_features=d, centers=mus,
                                cluster_std=[args.sigma, args.sigma])


            nb = GaussianNB()
            nb.fit(X_train, y_train)
            vars_nb = np.sum(nb.vars_groupby * nb.prior.reshape(-1,1), axis=0).reshape(1,-1)
                        
            for j, gamma in enumerate(args.gamma_list):

                aug_size = int(gamma * args.m)
                if aug_size > 0:
                    X_new, y_new = make_blobs(aug_size, n_features=d, centers=[nb.mus[0,:],nb.mus[1,:]],
                                                cluster_std=[np.sqrt(vars_nb), np.sqrt(vars_nb)])
                    X_aug = np.concatenate([X_train, X_new], axis=0)
                    y_aug = np.concatenate([y_train, y_new])
                else:
                    X_aug = X_train
                    y_aug = y_train

                theta_aug = np.sum(X_aug * (2 * y_aug-1).reshape(-1,1), axis=0) / len(y_aug)
                zero_one_loss_theta2_train = get_zero_one_loss(theta_aug, X_aug, y_aug)
                nll_loss_theta2_train = get_NLL_loss(theta_aug, X_aug, y_aug, d, args.sigma)
                zero_one_loss_theta2_test = get_zero_one_loss(theta_aug, X_test, y_test)
                nll_loss_theta2_test = get_NLL_loss(theta_aug, X_test, y_test, d, args.sigma)

                zero_one_loss_theta2_gap = np.abs(zero_one_loss_theta2_train - zero_one_loss_theta2_test)
                nll_loss_theta2_gap = np.abs(nll_loss_theta2_train - nll_loss_theta2_test)

                logger.info('t = %d, d = %d, gamma = %.2f, aug_size = %d, 01 loss train = %.4f, 01 loss test = %.4f, gap = %.4f' % \
                            (t+1, d, gamma, aug_size, zero_one_loss_theta2_train, zero_one_loss_theta2_test, zero_one_loss_theta2_gap))
                logger.info('t = %d, d = %d, gamma = %.2f, aug_size = %d, nll loss train = %.4f, nll loss test = %.4f, gap = %.4f' % \
                            (t+1, d, gamma, aug_size, nll_loss_theta2_train, nll_loss_theta2_test, nll_loss_theta2_gap))
                
                zero_one_loss_2[i,j,t] = zero_one_loss_theta2_gap
                nll_loss_2[i,j,t] = nll_loss_theta2_gap
            
    np.save(zero_one_loss_path_2, zero_one_loss_2)
    np.save(nll_loss_path_2, nll_loss_2)

if __name__ == '__main__':
    main()