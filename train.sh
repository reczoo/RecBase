# The code to train cl-vae
CUDA_VISIBLE_DEVICES=3 python -u main.py \
  --lr 1e-4 \
  --epochs 100 \
  --batch_size 2048 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --dropout_prob 0.0 \
  --bn False \
  --e_dim 128 \
  --quant_loss_weight 1.0 \
  --beta 0.25 \
  --num_emb_list 2048 2048 \
  --sk_epsilons 0.0 0.003 \
  --layers 2048 1024 512 256 128 64 \
  --device cuda \
  --data_path  'data_process/embeddings.npy'\
  --ckpt_dir ./ckpt/ \