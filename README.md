# GASDU
Install the required packages:

conda env create -f environment.yml

Example usage to run experiment on ARC-C:

cond activate gasdu

python train.py arc_c \
  --model_name "GPT-OSS-20B" \
  --finetune_method gasdu \
  --stage2_only \
  --max_epoch 3 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --update_percent 0.01 \
  --mask_mode topk \
  --full_grad_every 50
