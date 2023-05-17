export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:'pwd'
python main_train_aug.py --data-dir ./datasets \
    --log-dir ./log/RN18_cDCGAN_base \
    --desc RN18_cifar10s_lr0p2_epoch100_bs512_1000k \
    --data cifar10s \
    --batch-size 512 \
    --model resnet18 \
    --num-epochs 100 \
    --eval-freq 10 \
    --lr 0.2 \
    --augment base \
    --aux-data-filename ./datasets/cDCGAN/1000k.npz

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:'pwd'
python main_train_aug.py --data-dir ./datasets \
    --log-dir ./log/RN18_stylegan_base \
    --desc RN18_cifar10s_lr0p2_epoch100_bs512_100k \
    --data cifar10s \
    --batch-size 512 \
    --model resnet18 \
    --num-epochs 100 \
    --eval-freq 10 \
    --lr 0.2 \
    --augment base \
    --aux-data-filename ./datasets/stylegan/100k.npz
    
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:'pwd'
python main_train_aug.py --data-dir ./datasets \
    --log-dir ./log/RN18_stylegan \
    --desc RN18_cifar10s_lr0p2_epoch100_bs512_100k \
    --data cifar10s \
    --batch-size 512 \
    --model resnet18 \
    --num-epochs 100 \
    --eval-freq 10 \
    --lr 0.2 \
    --aux-data-filename ./datasets/stylegan/100k.npz


export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:'pwd'
python main_train_aug.py --data-dir ./datasets \
    --log-dir ./log/RN18_EDM \
    --desc RN18_cifar10s_lr0p2_epoch100_bs512_1000k \
    --data cifar10s \
    --batch-size 512 \
    --model resnet18 \
    --num-epochs 100 \
    --eval-freq 10 \
    --lr 0.2 \
    --aux-data-filename ../bishe/codes/data/5m.npz \
    --aux-take-amount 1000000