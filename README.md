# GASDU

A lightweight recipe for **Gradient–Southwell Dynamic Update (GASDU)** experiments.

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate gasdu
```
## Our experiments in GASDU paper run in two stages:
- stage1: 1-epoch grid search over batch size and learning rate
- stage2: 3-epoch fine-tuning using the best Stage 1 validation config

### run stage1 only:
```bash
python train.py arc_c \
  --model_name "GPT-OSS-20B" \
  --finetune_method gasdu \
  --stage1_only \
  --max_epoch 3 \
  --update_percent 0.01 \
  --mask_mode topk \
  --full_grad_every 50 # mask refresh period
```

### run stage2 only:

```bash
python train.py arc_c \
  --model_name "GPT-OSS-20B" \
  --finetune_method gasdu \
  --stage2_only \
  --max_epoch 3 \
  --batch_size 8 \ # specify batch size and lr for stage2
  --learning_rate 5e-5 \
  --update_percent 0.01 \
  --mask_mode topk \
  --full_grad_every 50 # mask refresh period
```

 ### run both:
  
  ```bash
  python train.py arc_c \
  --model_name "GPT-OSS-20B" \
  --finetune_method gasdu \
  --max_epoch 3 \
  --update_percent 0.01 \
  --mask_mode topk \
  --full_grad_every 50 # mask refresh period
```
