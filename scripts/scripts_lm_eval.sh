# Sample
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf  \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,parallelize=True  \
    --tasks wikitext  \
    --batch_size 8


# LLaMA-2-7B
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf  \
    --model_args pretrained=meta-llama/Llama-2-7b-hf  \
    --tasks lambada,openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,mmlu,gsm8k  \
    --batch_size 8 > results/lm_eval/llama2_7b.txt

CUDA_VISIBLE_DEVICES=1 lm_eval --model hf  \
    --model_args pretrained=q_llama2_7b_hf \
    --tasks lambada,openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,mmlu,gsm8k  \
    --batch_size 8 > results/lm_eval/q_llama2_7b_per_tensor_w1.0_a1.0.txt

# LLaMA-3-8B
CUDA_VISIBLE_DEVICES=2 lm_eval --model hf  \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B  \
    --tasks lambada,openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,mmlu,gsm8k  \
    --batch_size 8 > results/lm_eval/llama3_8b.txt

CUDA_VISIBLE_DEVICES=3 lm_eval --model hf  \
    --model_args pretrained=q_llama3_8b_hf  \
    --tasks lambada,openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,mmlu,gsm8k  \
    --batch_size 8 > results/lm_eval/q_llama3_8b_per_tensor_w1.0_a1.0.txt


triviaqa 3~4Hr
gsm8k -8shot 40min
mmlu 很快
minerva_math -4shot 2.5Hr
gpqa 2卡 2.5Hr


CUDA_VISIBLE_DEVICES=0 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --static_scales 'act_scales/llama2-7b-hf-static-weight.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --w_clip_ratio 1.0 --a_clip_ratio 1.0 \
    --quantize_output --skip_down_proj \
    --eval_ppl  --eval_common_sense | tee results/lm_eval/q_llama2_7b_per_tensor_w1.0_a1.0_new.txt

CUDA_VISIBLE_DEVICES=1 \
python model/main.py meta-llama/Llama-2-7b-hf wikitext2 \
    --smoothquant --static_scales 'act_scales/llama2-7b-hf-static-weight.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --w_clip_ratio 0.75 --a_clip_ratio 0.75 \
    --quantize_output --skip_down_proj \
    --eval_ppl  --eval_common_sense | tee results/lm_eval/q_llama2_7b_per_tensor_w0.75_a0.75_new.txt

CUDA_VISIBLE_DEVICES=2 \
python model/main.py meta-llama/Meta-Llama-3-8B wikitext2 \
    --smoothquant --static_scales 'act_scales/llama3-8b-hf-static-weight.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --w_clip_ratio 1.0 --a_clip_ratio 1.0 \
    --quantize_output --skip_down_proj \
    --eval_ppl  --eval_common_sense | tee results/lm_eval/q_llama2_7b_per_tensor_w1.0_a1.0_new.txt

CUDA_VISIBLE_DEVICES=3 \
python model/main.py meta-llama/Meta-Llama-3-8B wikitext2 \
    --smoothquant --static_scales 'act_scales/llama3-8b-hf-static-weight.pt' \
    --w_quant 'per_tensor' --a_quant 'per_tensor' \
    --w_clip_ratio 0.95 --a_clip_ratio 1.0 \
    --quantize_output --skip_down_proj \
    --eval_ppl  --eval_common_sense | tee results/lm_eval/q_llama2_7b_per_tensor_w0.95_a1.0_new.txt