XLA_FLAGS=--xla_gpu_cuda_data_dir=/raid/kirill_kirillov/.conda/envs/keras3/ \
CUDA_VISIBLE_DEVICES=1 \
python -m src.train \
    --dataset=hkr \
    --train_epochs=1000 \