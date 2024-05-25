# Dense
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-chat-hf wikitext2 \
    --eval_ppl

# SmoothQuant (per_tensor) (dynamic)
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-chat-hf wikitext2 \
    --smoothquant \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --w_clip_ratio 1.0 --a_clip_ratio 1.0 \
    --eval_ppl --skip_down_proj --quantize_output \
    --save_model q_llama2_7b_chat_dynamic_per_tensor.pt

# SmoothQuant (per_tensor) (static)
CUDA_VISIBLE_DEVICES=3 \
python model/main.py meta-llama/Llama-2-7b-chat-hf wikitext2 \
    --smoothquant --static_scales 'act_scales/llama2-7b-chat-hf-static-weight.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --w_clip_ratio 0.75 --a_clip_ratio 0.75 \
    --eval_ppl --skip_down_proj --quantize_output \
    --save_model q_llama2_7b_chat_static_per_tensor.pt

CUDA_VISIBLE_DEVICES=2 \
python model/main.py meta-llama/Meta-Llama-3-8B-Instruct wikitext2 \
    --smoothquant --static_scales 'act_scales/llama3-8b-chat-hf-static-weight.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --w_clip_ratio 0.95 --a_clip_ratio 1.0 \
    --skip_down_proj --quantize_output \
    --eval_ppl --save_model q_llama3_8b_chat_static_per_tensor.pt
