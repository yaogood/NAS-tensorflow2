MODEL=NAONet_A_36_cifar10
MODEL_DIR=exp/$MODEL

fixed_arc="0 7 1 15 2 8 1 7 0 6 1 7 0 8 4 7 1 5 0 7 0 7 1 13 0 6 0 14 0 9 1 10 0 14 2 6 1 11 0 7"

python train_cifar.py \
  --model_dir=$MODEL_DIR \
  --arch="$fixed_arc" \
  --use_aux_head \
  --cutout_size=16 | tee -a ./logs/train.log