# Dense
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-chat-hf wikitext2 \
    --eval_ppl

# SmoothQuant (per_tensor) (dynamic)
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-chat-hf wikitext2 \
    --smoothquant \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --eval_ppl  --skip_down_proj

# SmoothQuant (per_tensor) (static)
CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-chat-hf wikitext2 \
    --smoothquant --static_scales 'act_scales/llama2-7b-chat-hf-static.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --eval_ppl  --skip_down_proj