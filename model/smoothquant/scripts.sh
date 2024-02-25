python model/generate_act_scales.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --output-path act_scales/llama2-7b.pt \
    --num-samples 512 \
    --seq-len 4096 \
    --dataset-path wikitext2