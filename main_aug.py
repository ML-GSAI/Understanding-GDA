# generate samples by using cDCGAN and StyleGAN-V2
# and store them to npz files
import argparse
import os
import numpy as np

import torch
from models.cdcgan import Generator as CDCGAN

from utils import seed_torch
from utils import legacy

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./datasets')
parser.add_argument('--desc', type=str, default='cDCGAN', 
                    help='Name of model. It will be used to name directories.')
parser.add_argument('--sample_list', default=[1e5, 3e5, 5e5, 7e5, 1e6], type=list, nargs='+',
                    help='sample list')
parser.add_argument('--model', default='stylegan', type=str, choices=['CDCGAN', 'stylegan'],
                    help='pretrained generative model.')
parser.add_argument('--model-path', default='', type=str, help='path of pretrained generative model.')
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--width", type=int, default=128, help="number of feature maps")
parser.add_argument("--batch_size", type=int, default=500, help="size of the batches")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")

def main():
    args = parser.parse_args()
    print(args)

    args.store_dir = os.path.join(args.data_dir, args.desc)
    if os.path.exists(args.store_dir):
        raise RuntimeError('existing generated data, please check!')
    os.makedirs(args.store_dir)

    cuda = True if torch.cuda.is_available() else False
    args.device = 'cuda' if cuda else 'cup'
    seed_torch(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for sample_size in args.sample_list:
        if args.model == 'CDCGAN':
            sample_CDCGAN(sample_size, args)
        elif args.model == 'stylegan':
            sample_stylegan(sample_size, args)
        else:
            raise NotImplementedError('no such model!')

def sample_CDCGAN(sample_size, args):
    model = CDCGAN(
        n_classes=args.n_classes, 
        latent_dim=args.latent_dim,
        channels=args.channels,
        width=args.width
        ).to(args.device)
    
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    for _, param in model.named_parameters():
            param.requires_grad = False
    
    # label preprocess
    onehot = torch.zeros(args.n_classes, args.n_classes)
    onehot = onehot.scatter_(1, torch.LongTensor(list(range(args.n_classes))).view(args.n_classes,1), 1)
    onehot = onehot.view(args.n_classes, args.n_classes, 1, 1)

    sample_each_class = int(sample_size / args.n_classes)
    iter_each_class = int(sample_each_class / args.batch_size)

    total_imgs = []
    total_labels = []

    for c in tqdm(range(args.n_classes)):
        for i in range(iter_each_class):

            z = torch.randn((args.batch_size, args.latent_dim)).to(args.device).float()
            z = z.view(-1, args.latent_dim, 1, 1)

            gen_labels = np.array([c] * args.batch_size)
            gen_labels_onehot = onehot[gen_labels].to(args.device)

            with torch.no_grad():
                gen_imgs = model(z, gen_labels_onehot)
            gen_imgs = gen_imgs * 0.5 + 0.5
            gen_imgs = gen_imgs.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1)
            gen_imgs = gen_imgs.detach().to("cpu", torch.uint8).numpy()
            
            total_imgs.append(gen_imgs)
            total_labels.append(gen_labels)
    
    total_imgs = np.concatenate(total_imgs)
    total_labels = np.concatenate(total_labels)

    permutation = np.random.permutation(total_labels.shape[0])
    total_imgs = total_imgs[permutation]
    total_labels = total_labels[permutation]

    file_name = str(int(sample_size / 1000)) + 'k'
    file_path = os.path.join(args.store_dir, file_name+'.npz')
    np.savez(file_path, image=total_imgs, label=total_labels)

def sample_stylegan(sample_size, args):
    f = open(args.model_path, 'rb')
    model = legacy.load_network_pkl(f)['G_ema'].to(args.device) # type: ignore
    for _, param in model.named_parameters():
            param.requires_grad = False
    # print(model)

    sample_each_class = int(sample_size / args.n_classes)
    iter_each_class = int(sample_each_class / args.batch_size)

    total_imgs = []
    total_labels = []

    for c in tqdm(range(args.n_classes)):
        for i in range(iter_each_class):
            # Labels.
            gen_labels = np.array([c] * args.batch_size)
            label = torch.zeros([args.batch_size, model.c_dim], device=args.device)
            label[:, c] = 1

            z = torch.randn(args.batch_size, model.z_dim).to(args.device)
            img = model(z, label, truncation_psi=1, noise_mode='random')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img.detach().to("cpu", torch.uint8).numpy()

            total_imgs.append(img)
            total_labels.append(gen_labels)
    
    total_imgs = np.concatenate(total_imgs)
    total_labels = np.concatenate(total_labels)

    permutation = np.random.permutation(total_labels.shape[0])
    total_imgs = total_imgs[permutation]
    total_labels = total_labels[permutation]

    file_name = str(int(sample_size / 1000)) + 'k'
    file_path = os.path.join(args.store_dir, file_name+'.npz')
    np.savez(file_path, image=total_imgs, label=total_labels)


if __name__ == '__main__':
    main()