#!/bin/bash
# GRPO 纯规则奖励复现（在 MiniMind/trainer 目录下运行）
# 前置：1) 已下载 dataset/rlaif.jsonl（modelscope gongjy/minimind_dataset）
#       2) out/full_sft_768.pth 已存在（GRPO 从 SFT 权重热启动）
#       3) trainer/train_grpo_rule.py = 官方 train_grpo.py 去掉学习型 reward model
#          （把 LMForRewardModel(...) 一行替换为返回 0 分的 _DummyRuleReward）
cd /root/autodl-tmp/MiniMind/trainer
source /root/miniconda3/etc/profile.d/conda.sh
conda activate minimind
python train_grpo_rule.py \
  --data_path ../dataset/rlaif_sub.jsonl `# 600 条等间隔抽样子集` \
  --from_weight full_sft `# policy/reference 均从 full_sft 热启动` \
  --batch_size 2 --num_generations 4 `# 每 prompt 在线采样 4 条做组内比较` \
  --max_seq_len 512 --max_gen_len 384 `# 官方默认 768/1024，单卡适当缩短以控时长` \
  --epochs 1 --learning_rate 3e-7 --beta 0.1 \
  --loss_type cispo --thinking_ratio 0.9 \
  --num_workers 4 --log_interval 5 --save_interval 100 \
  --device cuda:0