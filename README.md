# Toward Understanding Generative Data Augmentation

## Dependencies

```bash
conda env create -f pytorch.yaml
```

## Simulation experiments on bGMM

* Reproduce the simulation results in Figure. 1

```bash
bash scripts/main_bGMM.sh
```

* Use the code in plot.ipynb to plot the results

## Empirical experiments on CIFAR-10

### Obtain weights of deep generative models

* cDCGAN: 

  ```bash
  python ./main_train_CDCGAN.py # hyperparameters have been set
  ```

* StyleGANV2-ADA: 

  ```bash
  wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl
  ```

### Generate and store new images

* cDCGAN and StyleGAN2-ADA:

  ```bash
  bash scripts/main_aug.sh
  ```

* EDM:

  ```bash
  wget https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/5m.npz
  ```

### Train ResNets with GDA (see  scripts/main_train_aug.sh)

* cDCGAN and StyleGAN2-ADA:

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=$PYTHONPATH:'pwd'
  python main_train_aug.py --data-dir ./datasets \
      --log-dir ./log/RN18_cDCGAN_base \
      --desc RN18_cifar10s_lr0p2_epoch100_bs512_1000k \ # dependent on m_G, here m_G = 1M
      --data cifar10s \
      --batch-size 512 \
      --model resnet18 \
      --num-epochs 100 \
      --eval-freq 10 \
      --lr 0.2 \
      --aux-data-filename ./datasets/cDCGAN/1000k.npz # dependent on m_G, here m_G = 1M
      --augment base \ # none if do not use standard augmentation
  ```

* EDM:

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONPATH=$PYTHONPATH:'pwd'
  python main_train_aug.py --data-dir ./datasets \
      --log-dir ./log/RN18_EDM_base \
      --desc RN18_cifar10s_lr0p2_epoch100_bs512_1000k \
      --data cifar10s \
      --batch-size 512 \
      --model resnet18 \
      --num-epochs 100 \
      --eval-freq 10 \
      --lr 0.2 \
      --aux-data-filename ../bishe/codes/data/5m.npz \ # dir of the downloaded EDM data
      --aux-take-amount 1000000 # m_G
      --augment base \ # none if do not use standard augmentation
  ```

## Acknowledgments

The code is developed based on the following repositories. We appreciate their nice implementations.

|       Method        |                          Repository                          |
| :-----------------: | :----------------------------------------------------------: |
|       CDCGAN        |  https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN  |
|      StyleGAN       |       https://github.com/NVlabs/stylegan2-ada-pytorch        |
| EDM data & training |          https://github.com/wzekai99/DM-Improves-AT          |
|        bGMM         | https://github.com/ML-GSAI/Revisiting-Dis-vs-Gen-Classifiers |