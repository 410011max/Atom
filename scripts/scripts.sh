# Sample
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --wbits 4 --abits 4 --a_sym --w_sym \
    --act_group_size 128 --weight_group_size 128 --weight_channel_group 2 \
    --reorder --act_sort_metric hessian \
    --a_clip_ratio 0.9 --w_clip_ratio 0.85 \
    --keeper 128 --keeper_precision 3 --kv_cache --use_gptq \
    --eval_ppl --eval_common_sense

##################################################################

# W16A16
CUDA_VISIBLE_DEVICES=0 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --eval_ppl

##################################################################

# SmoothQuant (per_tensor)
CUDA_VISIBLE_DEVICES=0 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --act_scales 'act_scales/llama2-7b-hf.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' --alpha 0.85 \
    --eval_ppl

# SmoothQuant (per_channel)
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --act_scales 'act_scales/llama2-7b-hf.pt' \
    --w_quant 'per_channel' --a_quant 'per_token' --alpha 0.85 \
    --eval_ppl

##################################################################

# Atom W8A8 per-channel
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --wbits 8 --abits 8 --a_sym --w_sym \
    --a_clip_ratio 0.9 --w_clip_ratio 0.85 \
    --use_gptq --eval_ppl

# Atom W4A8 per-channel
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --wbits 4 --abits 8 --a_sym --w_sym \
    --a_clip_ratio 0.9 --w_clip_ratio 0.85 \
    --use_gptq --eval_ppl

# Atom W4A8 without reordering
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --wbits 4 --abits 8 --a_sym --w_sym \
    --act_group_size 128 --weight_group_size 128 --weight_channel_group 1 \
    --a_clip_ratio 0.9 --w_clip_ratio 0.85 \
    --use_gptq --eval_ppl

# Atom W8A8 without reordering
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --wbits 8 --abits 8 --a_sym --w_sym \
    --act_group_size 128 --weight_group_size 128 --weight_channel_group 1 \
    --a_clip_ratio 0.9 --w_clip_ratio 0.85 \
    --use_gptq --eval_ppl

# Atom W8A8 with reordering
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --wbits 8 --abits 8 --a_sym --w_sym \
    --act_group_size 128 --weight_group_size 128 --weight_channel_group 1 \
    --reorder --act_sort_metric hessian \
    --a_clip_ratio 0.9 --w_clip_ratio 0.85 \
    --keeper 0 --keeper_precision 3 --use_gptq \
    --eval_ppl
