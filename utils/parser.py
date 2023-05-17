import argparse

from data import DATASETS
from models import MODELS
from .train import SCHEDULERS

from .tools import str2bool, str2float


def parser_train():
    """
    Parse input arguments (train.py).
    """
    parser = argparse.ArgumentParser(description='Standard Training.')

    parser.add_argument('--augment', type=str, default='none', choices=['none', 'base', 'cutout', 'autoaugment', 'randaugment', 'idbh'], help='Augment training set.')

    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--batch-size-validation', type=int, default=256, help='Batch size for testing.')
    
    parser.add_argument('--data-dir', type=str, default='./datasets')
    parser.add_argument('--log-dir', type=str, default='./log')
    
    parser.add_argument('--data', type=str, default='cifar10s', choices=DATASETS, help='Data to use.')
    parser.add_argument('--desc', type=str, required=True, 
                        help='Description of experiment. It will be used to name directories.')

    parser.add_argument('--model', choices=MODELS, default='resnet18', help='Model architecture to be used.')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
    parser.add_argument('--pretrained-file', type=str, default=None, help='Pretrained weights file name.')

    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--eval-freq', type=int, default=10, help='evaluation frequency (in epochs).')
    
    parser.add_argument('--beta', default=None, type=float, help='Stability regularization, i.e., 1/lambda in TRADES.')
    
    parser.add_argument('--lr', type=float, default=0.2, help='Learning rate for optimizer (SGD).')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='cosinew', help='Type of scheduler.')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='Use Nesterov momentum.')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient norm clipping.')

    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Debug code. Run 1 epoch of training and evaluation.')
    
    parser.add_argument('--unsup-fraction', type=float, default=0.7, help='Ratio of unlabelled data to labelled data.')
    parser.add_argument('--aux-data-filename', type=str, help='Path to additional Tiny Images data.', 
                        default=None)
    parser.add_argument('--aux-take-amount', type=int, default=None, help='Number of augmentation.')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    ### Consistency
    parser.add_argument('--consistency', action='store_true', default=False, help='use Consistency.')
    parser.add_argument('--cons_lambda', type=float, default=1.0, help='lambda for Consistency.')
    parser.add_argument('--cons_tem', type=float, default=0.5, help='temperature for Consistency.')

    ### Resume
    parser.add_argument('--resume_path', default='', type=str)

    ### Our methods
    parser.add_argument('--LSE', action='store_true', default=False, help='LSE training.')
    parser.add_argument('--ls', type=float, default=0., help='label smoothing.')
    parser.add_argument('--clip_value', default=0, type=float)
    parser.add_argument('--CutMix', action='store_true', default=False, help='use CutMix.')
    return parser


def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description='Robustness evaluation.')

    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/scratch/rarade/test/')
        
    parser.add_argument('--desc', type=str, required=True, help='Description of model to be evaluated.')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples.')

    # eval-aa.py
    parser.add_argument('--train', action='store_true', default=False, help='Evaluate on training set.')
    parser.add_argument('-v', '--version', type=str, default='standard', choices=['custom', 'plus', 'standard'], 
                        help='Version of AA.')

    # eval-adv.py
    parser.add_argument('--source', type=str, default=None, help='Path to source model for black-box evaluation.')
    parser.add_argument('--wb', action='store_true', default=False, help='Perform white-box PGD evaluation.')
    
    # eval-rb.py
    parser.add_argument('--threat', type=str, default='corruptions', choices=['corruptions', 'Linf', 'L2'],
                        help='Threat model for RobustBench evaluation.')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    return parser

