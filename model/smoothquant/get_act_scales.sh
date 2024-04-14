CUDA_VISIBLE_DEVICES=0 \
python model/generate_act_scales.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --mode smooth \
    --output-path act_scales/llama2-7b-hf-smooth.pt \
    --num-samples 512 \
    --seq-len 2048

CUDA_VISIBLE_DEVICES=1 \
python model/generate_act_scales.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --mode static \
    --output-path act_scales/llama2-7b-hf-static-weight.pt \
    --num-samples 512 \
    --seq-len 2048

CUDA_VISIBLE_DEVICES=1 \
python model/generate_act_scales.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --mode static \
    --output-path act_scales/llama2-7b-hf-smooth-static.pt \
    --num-samples 512 \
    --seq-len 2048 \
    --smooth-scales 'act_scales/llama2-7b-hf-smooth.pt' \
    --alpha 0.85

