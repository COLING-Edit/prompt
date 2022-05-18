# UPSA_pro

## Sentiment

### Non-finetuned PLM classifier
CUDA_VISIBLE_DEVICES=0 python3 run_sa.py --style_weight 3 \
--direction 1-0 \
--style_mode plm \
--task sentiment \
--class_name EleutherAI/gpt-neo-2.7B \
--topk 50 \
--max_steps 15 \
--output_dir plm/ \
--semantic_mode kw-sent \
--keyword_weight 8 \
--sent_weight 1 \
--max_len 16 \
--action all \
--early_stop True

## Formality
CUDA_VISIBLE_DEVICES=1 python3 run_sa.py --style_weight 8 \
--direction 0-1 \
--style_mode plm \
--task formality \
--class_name EleutherAI/gpt-neo-2.7B \
--topk 50 \
--max_steps 20 \
--output_dir formality/ \
--semantic_mode kw-sent \
--keyword_weight 1 \
--fluency_weight 4 \
--sent_weight 1 \
--max_len 23 \
--action all \
--early_stop True

perl multi-bleu.perl ../data/GYAFC_500/pos2neg_ref/ref0.0 ../data/GYAFC_500/pos2neg_ref/ref1.0 ../data/GYAFC_500/pos2neg_ref/ref2.0 ../data/GYAFC_500/pos2neg_ref/ref3.0 < ../output/gpt3-babbage-001_1-0.txt
perl multi-bleu.perl ../data/GYAFC_500/neg2pos_ref/ref0.1 ../data/GYAFC_500/neg2pos_ref/ref1.1 ../data/GYAFC_500/neg2pos_ref/ref2.1 ../data/GYAFC_500/neg2pos_ref/ref3.1 < ../output/gpt3-babbage-001_0-1.txt

perl multi-bleu.perl ../data/yelp/references/reference0 ../data/yelp/references/reference1 ../data/yelp/references/reference2 ../data/yelp/references/reference3 < ../output/sentiment_both.txt
perl multi-bleu.perl ../data/gyafc_500/references/ref0 ../data/gyafc_500/references/ref1 ../data/gyafc_500/references/ref2 ../data/gyafc_500/references/ref3 < ../output/gyafc_both.txt

perl multi-bleu.perl ../data/yelp/pos2neg_ref/ref0.1 ../data/yelp/pos2neg_ref/ref1.1 ../data/yelp/pos2neg_ref/ref2.1 ../data/yelp/pos2neg_ref/ref3.1 < ../output/gpt3-babbage-001_senti_1-0.txt
perl multi-bleu.perl ../data/yelp/neg2pos_ref/ref0.0 ../data/yelp/neg2pos_ref/ref1.0 ../data/yelp/neg2pos_ref/ref2.0 ../data/yelp/neg2pos_ref/ref3.0 < ../output/gpt3-babbage-001_senti_0-1.txt

perl multi-bleu.perl ../data/yelp/references/reference0.1 ../data/yelp/references/reference1.1 ../data/yelp/references/reference2.1 ../data/yelp/references/reference3.1 < ../output.txt
perl multi-bleu.perl ../data/yelp/references/reference0.0 ../data/yelp/references/reference1.0 ../data/yelp/references/reference2.0 ../data/yelp/references/reference3.0 < ../output.txt