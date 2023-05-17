import json
import time
import argparse
import shutil

from tqdm import tqdm

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from data import get_data_info
from data import load_data
from data import SEMISUP_DATASETS

from utils import format_time
from utils import Logger
from utils import parser_train
from utils import Trainer
from utils import seed_torch

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Setup

parse = parser_train()
args = parse.parse_args()
assert args.data in SEMISUP_DATASETS, f'Only data in {SEMISUP_DATASETS} is supported!'


DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)


info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
NUM_EPOCHS = args.num_epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))
if args.debug:
    NUM_EPOCHS = 1

# To speed up training and fix random seed
seed_torch(args.seed)


# Load data
train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, use_consistency=args.consistency, shuffle_train=True, 
    aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, aux_take_amount=args.aux_take_amount, validation=False
)
print(len(train_dataset))
del train_dataset, test_dataset

trainer = Trainer(info, args)
last_lr = args.lr


if NUM_EPOCHS > 0:
    logger.log('\n\n')
    metrics = pd.DataFrame()
    logger.log('Standard Accuracy-\tEval: {:2f}%.'.format(trainer.eval(test_dataloader)[1]*100))
    
    old_score = 0.0
    logger.log('Standard training for {} epochs'.format(NUM_EPOCHS))
    trainer.init_optimizer(args.num_epochs)
    eval_acc = 0.0

if args.resume_path:
    start_epoch = trainer.load_model_resume(os.path.join(args.resume_path, 'state-last.pt')) + 1
    logger.log(f'Resuming at epoch {start_epoch}')
else:
    start_epoch = 1

for epoch in tqdm(range(start_epoch, NUM_EPOCHS+1)):
    start = time.time()
    logger.log('======= Epoch {} ======='.format(epoch))
    
    if args.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]
    
    res = trainer.train(train_dataloader, epoch=epoch, adversarial=False)
    
    logger.log('Loss: {:.4f}.\tLR: {:.4f}'.format(res['loss'], last_lr))
    logger.log('Mean Accuracy-\tTrain: {:.2f}%.'.format(res['clean_acc']*100))

    epoch_metrics = {'train_'+k: v for k, v in res.items()}
    # epoch_metrics.update({'epoch': epoch, 'lr': last_lr, 'test_clean_acc': test_acc, 'test_adversarial_acc': ''})
    epoch_metrics.update({'epoch': epoch, 'lr': last_lr})

    if epoch % args.eval_freq == 0 or epoch == NUM_EPOCHS:
        train_loss, train_acc = trainer.eval(train_dataloader, adversarial=False) 
        eval_loss, eval_acc = trainer.eval(test_dataloader, adversarial=False)
        logger.log('Loss-\tTrain: {:.4f}.\tTest: {:.4f}.\tGap: {:.4f}.'.format(train_loss, eval_loss, np.abs(train_loss-eval_loss)))
        logger.log('Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.\tGap: {:.2f}%.'.format(train_acc*100, eval_acc*100, np.abs(train_acc*100-eval_acc*100)))
        epoch_metrics.update({'train_acc': train_acc*100})
        epoch_metrics.update({'eval_acc': eval_acc*100})
        epoch_metrics.update({'train_loss2': train_loss})
        epoch_metrics.update({'eval_loss2': eval_loss})
        epoch_metrics.update({'acc_gap': np.abs(train_acc*100-eval_acc*100)})
        epoch_metrics.update({'loss_gap': np.abs(train_loss-eval_loss)})

        trainer.save_model(os.path.join(LOG_DIR, 'state-last.pt')) 

    
    if eval_acc > old_score:
        old_score = eval_acc
        trainer.save_model(WEIGHTS)

    if epoch % NUM_EPOCHS == 0:
        shutil.copyfile(WEIGHTS, os.path.join(LOG_DIR, f'weights-best-epoch{str(epoch)}.pt'))

    logger.log('Time taken: {}'.format(format_time(time.time()-start)))
    metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
    metrics.to_csv(os.path.join(LOG_DIR, 'stats.csv'), index=False)

    
    
# Record metrics
logger.log('\nTraining completed.')
logger.log('Standard Accuracy-\tBest Test: {:.2f}%.'.format(old_score*100))
logger.log('Script Completed.')