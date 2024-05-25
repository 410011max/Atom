##################################################################
# W16A16
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --eval_ppl --eval_common_sense

CUDA_VISIBLE_DEVICES=0 \
python model/main.py meta-llama/Meta-Llama-3-8B wikitext2 \
    --eval_ppl  --eval_common_sense

##################################################################

# Mx
CUDA_VISIBLE_DEVICES=2 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --mx --mx_format fp6_e3m2 --mx_block_size 32 --eval_ppl --seqlen 2048

CUDA_VISIBLE_DEVICES=0 \
python model/main.py meta-llama/Meta-Llama-3-8B wikitext2 \
    --mx --mx_format int8 --mx_block_size 32 --eval_ppl --seqlen 2048

# Mx (smooth)
CUDA_VISIBLE_DEVICES=3 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smooth_scales 'act_scales/llama2-7b-hf.pt' --alpha 0.85 \
    --mx --mx_format int8 --mx_block_size 32 --eval_ppl --seqlen 2048
