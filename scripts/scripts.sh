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
    --eval_ppl --eval_common_sense

##################################################################
# region mx
# Mx
CUDA_VISIBLE_DEVICES=2 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --mx --mx_format int8 --mx_block_size 32 --eval_ppl --seqlen 1024

# Mx (smooth)
CUDA_VISIBLE_DEVICES=3 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smooth_scales 'act_scales/llama2-7b-hf.pt' --alpha 0.85 \
    --mx --mx_format int8 --mx_block_size 32 --eval_ppl --seqlen 1024

# endregion
##################################################################
# region dynamic
# SmoothQuant (per_tensor) (dynamic)
CUDA_VISIBLE_DEVICES=3 \
python model/main.py meta-llama/Llama-2-7b-chat-hf wikitext2 \
    --smoothquant \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --eval_ppl --quantize_output --skip_down_proj \
    --save_model 'q_llama2_7b_chat_dynamic_per_tensor.pt'

# SmoothQuant (per_channel + per_tensor) (dynamic)
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant \
    --w_quant 'per_channel' --a_quant 'per_tensor' \
    --eval_ppl --quantize_output --skip_down_proj

# SmoothQuant (per_channel + per_token) (dynamic)
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-chat-hf wikitext2 \
    --smoothquant \
    --w_quant 'per_channel' --a_quant 'per_token' \
    --eval_ppl --quantize_output --skip_down_proj

# SmoothQuant (per_tensor) (dynamic) (smooth)
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --smooth_scales 'act_scales/llama2-7b-hf-smooth.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' --alpha 0.85 \
    --eval_ppl --quantize_output --skip_down_proj

# SmoothQuant (per_channel) (dynamic) (smooth)
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --smooth_scales 'act_scales/llama2-7b-hf-smooth.pt' \
    --w_quant 'per_channel' --a_quant 'per_token' --alpha 0.85 \
    --eval_ppl

# endregion

# region static

# SmoothQuant (per_tensor) (static)
CUDA_VISIBLE_DEVICES=3 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --static_scales 'act_scales/llama2-7b-hf-static-weight-clip_0.75.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --w_clip_ratio 1.0 --a_clip_ratio 1.0 \
    --eval_ppl --quantize_output --skip_down_proj \
    --eval_common_sense 

# SmoothQuant (per_8_channel) (static)
CUDA_VISIBLE_DEVICES=2 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --static_scales 'act_scales/llama2-7b-hf-static.pt' \
    --w_quant 'per_8_channel' --a_quant 'per_tensor' \
    --eval_ppl --quantize_output --skip_down_proj   

# SmoothQuant (per_channel) (static)
CUDA_VISIBLE_DEVICES=2 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --static_scales 'act_scales/llama2-7b-hf-static.pt' \
    --w_quant 'per_channel' --a_quant 'per_tensor' \
    --eval_ppl --quantize_output --skip_down_proj

# SmoothQuant (per_tensor) (static) (smooth)
CUDA_VISIBLE_DEVICES=0 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --smooth_scales 'act_scales/llama2-7b-hf-smooth.pt' \
    --static_scales 'act_scales/llama2-7b-hf-smooth-static.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --eval_ppl --quantize_output

# SmoothQuant (per_token) (static) (smooth)
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --smooth_scales 'act_scales/llama2-7b-hf-smooth.pt' \
    --static_scales 'act_scales/llama2-7b-hf-smooth-static.pt' \
    --w_quant 'per_channel' --a_quant 'per_token' \
    --eval_ppl --quantize_output

# endregion

##################################################################
# region Atom
# Atom W8A8 per-channel
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --wbits 8 --abits 8 --a_sym --w_sym --static\
    --a_clip_ratio 0.9 --w_clip_ratio 0.85 \
    --use_gptq --eval_ppl

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

# Atom W4A4 in paper
CUDA_VISIBLE_DEVICES=2 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --wbits 4 --abits 4 --a_sym --w_sym \
    --act_group_size 128 --weight_group_size 128 --weight_channel_group 2 \
    --reorder --act_sort_metric hessian \
    --a_clip_ratio 0.9 --w_clip_ratio 0.85 \
    --keeper 128 --keeper_precision 3 --kv_cache --use_gptq \
    --eval_ppl
# endregion