export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`

python main_aug.py \
  --desc CDCGAN \
  --model CDCGAN \
  --model-path "./log/cDCGAN/G-weights-last.pt"

python main_aug.py \
  --desc stylegan \
  --model stylegan \
  --model-path ./log/stylegan/cifar10.pkl