# GASDU

A lightweight recipe for **Gradient–Southwell Dynamic Update (GASDU)** experiments.

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate gasdu
```
### GASDU training runs in two stages:
- stage1: 1-epoch grid search over batch size and learning rate
- stage2: 3-epoch fine-tuning using the best Stage 1 validation config

## Quick Start: ARC-C
## run stage1 only:
```bash
python train.py arc_c \
  --model_name "GPT-OSS-20B" \
  --finetune_method gasdu \
  --stage1_only \
  --max_epoch 3 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --update_percent 0.01 \
  --mask_mode topk \
  --full_grad_every 50
```

## run stage2 only:

```bash
python train.py arc_c \
  --model_name "GPT-OSS-20B" \
  --finetune_method gasdu \
  --stage1_only \
  --max_epoch 3 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --update_percent 0.01 \
  --mask_mode topk \
  --full_grad_every 50
```

 ## run both:
  
  ```bash
  python train.py arc_c \
  --model_name "GPT-OSS-20B" \
  --finetune_method gasdu \
  --max_epoch 3 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --update_percent 0.01 \
  --mask_mode topk \
  --full_grad_every 50
```
