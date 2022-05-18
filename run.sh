#!/bin/bash

python3 SAHC.py \
--style_weight 12 \
--direction 0-1 \
--style_mode plm \
--class_name EleutherAI/gpt-j-6B \
--topk 40 \
--max_steps 2 \
--output_dir sentiment/ \
--task sentiment \
--semantic_mode kw-sent \
--keyword_weight 1 \
--sent_weight 1 \
--max_len 16 \
--action all \
--early_stop True \
--fluency_weight 8 \
--bleu_weight 1 \
--seed 42 \
--setting zero-shot

python3 SAHC.py \
--style_weight 12 \
--direction 1-0 \
--style_mode plm \
--class_name EleutherAI/gpt-j-6B \
--topk 40 \
--max_steps 2 \
--output_dir sentiment/ \
--task sentiment \
--semantic_mode kw-sent \
--keyword_weight 1 \
--sent_weight 1 \
--max_len 16 \
--action all \
--early_stop True \
--fluency_weight 8 \
--bleu_weight 1 \
--seed 42 \
--setting zero-shot

python3 SAHC.py \
--style_weight 12 \
--direction 0-1 \
--style_mode plm \
--class_name EleutherAI/gpt-j-6B \
--topk 20 \
--max_steps 5 \
--output_dir formality/ \
--task formality \
--semantic_mode kw-sent \
--keyword_weight 1 \
--sent_weight 1 \
--max_len 24 \
--action all \
--early_stop True \
--fluency_weight 4 \
--bleu_weight 1 \
--seed 42 \
--setting zero-shot

python3 SAHC.py \
--style_weight 12 \
--direction 1-0 \
--style_mode plm \
--class_name EleutherAI/gpt-j-6B \
--topk 20 \
--max_steps 5 \
--output_dir formality/ \
--task formality \
--semantic_mode kw-sent \
--keyword_weight 1 \
--sent_weight 1 \
--max_len 24 \
--action all \
--early_stop True \
--fluency_weight 4 \
--bleu_weight 1 \
--seed 42 \
--setting zero-shot