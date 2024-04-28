

# LLaMA-3-8B
for w_clip in 1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6; do
    for a_clip in 1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6; do
        CUDA_VISIBLE_DEVICES=0 \
        python model/main.py meta-llama/Meta-Llama-3-8B wikitext2 \
            --smoothquant --static_scales 'act_scales/llama3-8b-hf-static-weight.pt' \
            --w_quant 'per_tensor' --a_quant 'per_tensor' \
            --w_clip_ratio $w_clip --a_clip_ratio $a_clip \
            --quantize_output --skip_down_proj \
            --eval_ppl > results/clipping/llama3_8b_w${w_clip}_a${a_clip}.txt
    done
done

# LLaMA-2-7B
for w_clip in 1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6; do
    for a_clip in 1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6; do
        CUDA_VISIBLE_DEVICES=0 \
        python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
            --smoothquant --static_scales 'act_scales/llama2-7b-hf-static-weight-clip_0.75.pt' \
            --w_quant 'per_tensor' --a_quant 'per_tensor' \
            --w_clip_ratio $w_clip --a_clip_ratio $a_clip \
            --quantize_output --skip_down_proj \
            --eval_ppl > results/clipping/llama2_7b_w${w_clip}_a${a_clip}.txt
    done
done