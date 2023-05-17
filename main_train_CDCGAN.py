import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import json
import time

import numpy as np

from models.cdcgan import Generator, Discriminator

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

import torch

from utils import Logger, seed_torch, format_time, weights_init_normal

def sample_image_grid(generator, z, n_row, batches_done, img_dir):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    y_label = onehot[labels].to(device)
    # generator.eval()
    with torch.no_grad():
        gen_imgs = generator(z, y_label)
    # generator.train()
    gen_imgs = gen_imgs * 0.5 + 0.5
    image_name = "%d.png" % batches_done
    image_path = os.path.join(img_dir, image_name)
    save_image(gen_imgs.data, image_path, nrow=n_row, normalize=True)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/cifar')
parser.add_argument('--log-dir', type=str, default='./log')
parser.add_argument('--desc', type=str, default='cDCGAN', 
                    help='Description of experiment. It will be used to name directories.')
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--width", type=int, default=128, help="number of feature maps")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
args = parser.parse_args()
print(args)

LOG_DIR = os.path.join(args.log_dir, args.desc)
IMAGE_DIR = os.path.join(LOG_DIR, "images")
D_WEIGHTS = os.path.join(LOG_DIR, 'D-weights-last.pt')
G_WEIGHTS = os.path.join(LOG_DIR, 'G-weights-last.pt')

if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
os.makedirs(IMAGE_DIR)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)


cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cup'
logger.log('If use cuda: {}'.format(cuda))

seed_torch(args.seed)

fixed_z = torch.randn((args.n_classes ** 2, args.latent_dim)).to(device).float()
fixed_z = fixed_z.view((-1, args.latent_dim, 1, 1))

# Loss functions
adversarial_loss = torch.nn.BCELoss().to(device)

# Initialize generator and discriminator
generator = Generator(
    n_classes=args.n_classes, 
    latent_dim=args.latent_dim,
    channels=args.channels,
    width=args.width
    ).to(device)

discriminator = Discriminator(
    n_classes=args.n_classes, 
    channels=args.channels,
    width=args.width
    ).to(device)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
train_transform = transforms.Compose([
             transforms.ToTensor(), 
             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
             ])
dataloader = DataLoader(
    datasets.CIFAR10(
        args.data_dir,
        train=True,
        download=True,
        transform=train_transform
    ),
    batch_size=args.batch_size,
    shuffle=True,
)
print(len(dataloader))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
fill = torch.zeros([10, 10, args.img_size, args.img_size])
for i in range(args.n_classes):
    fill[i, i, :, :] = 1

# ----------
#  Training
# ----------

logger.log('Standard training for {} epochs'.format(args.n_epochs))

for epoch in range(args.n_epochs):
    start = time.time()
    logger.log('======= Epoch {} ======='.format(epoch+1))

    d_loss_avg = 0
    g_loss_avg = 0
    
    for i, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = torch.ones((batch_size), requires_grad=False).to(device)
        fake = torch.zeros((batch_size), requires_grad=False).to(device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn((batch_size, args.latent_dim)).to(device).float()
        z = z.view(-1, args.latent_dim, 1, 1)
        gen_labels = np.random.randint(0, args.n_classes, batch_size)
        gen_labels_onehot = onehot[gen_labels].to(device)
        gen_labels_fill = fill[gen_labels].to(device)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels_onehot)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels_fill).squeeze()
        g_loss = adversarial_loss(validity, valid)
        g_loss_avg += g_loss.item()

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        labels_fill = fill[labels].to(device)
        # Loss for real images
        real_pred = discriminator(imgs, labels_fill).squeeze()
        d_real_loss = adversarial_loss(real_pred, valid) 

        # Loss for fake images
        fake_pred = discriminator(gen_imgs.detach(), gen_labels_fill).squeeze()
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss_avg += d_loss.item()

        d_loss.backward()
        optimizer_D.step()
    
    d_loss_avg /= len(dataloader)
    g_loss_avg /= len(dataloader)

    logger.log(
        "[Epoch %d/%d] [D loss: %.3f] [G loss: %.3f]"
        % (epoch + 1, args.n_epochs, d_loss_avg, g_loss_avg)
    )
    logger.log('Time taken: {}'.format(format_time(time.time()-start)))

    if (epoch + 1) % args.sample_interval == 0:
        sample_image_grid(generator, fixed_z, n_row=10, batches_done=epoch+1, img_dir=IMAGE_DIR)
        torch.save(generator.state_dict(), G_WEIGHTS)
        torch.save(discriminator.state_dict(), D_WEIGHTS)

logger.log('\nTraining completed.')
logger.log('Script Completed.')