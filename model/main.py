import os
import random
import numpy as np
import torch
from quant import *
from outlier import *
from eval import *
from collections import defaultdict
from pprint import pprint
from modelutils_llama import quantize_model_llama, reorder_model_llama, quantize_model_gptq_llama,  add_act_quant_wrapper_llama
from modelutils_opt import quantize_model_opt, reorder_model_opt, quantize_model_gptq_opt,  add_act_quant_wrapper_opt
from parallel_utils import map_layers_to_multi_gpus
from LMClass import LMClass
from eval import pattern_match
from lm_eval import tasks as lm_tasks
from lm_eval import evaluator as lm_evaluator

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_llama(args):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32 if args.mx else torch.float16
    )
    model.seqlen = args.seqlen
    return model

def get_opt(args):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32 if args.mx else torch.float16
    )
    model.seqlen = model.config.max_position_embeddings
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--seqlen', type=int, default=2048,
        help='Sequence length for the model.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, 
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    # Quantization Method
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantizing weight; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--abits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantizing activation; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--exponential', action='store_true',
        help='Whether to use exponential-only for weight quantization.'
    )
    parser.add_argument(
        '--a_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--w_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--static', action='store_true',
        help='Whether to perform static quantization (For activtions). Default is dynamic. (Deprecated in Atom)'
    )
    parser.add_argument(
        '--weight_group_size', type=int, default=0, choices=[0, 32, 64, 128, 256, 384, 768],
        help='Group size when quantizing weights. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--weight_channel_group', type=int, default=1,
        help='Group size of channels that will quantize together. (only for weights now)'
    )
    parser.add_argument(
        '--act_group_size', type=int, default=0, choices=[0, 64, 128, 256, 384, 768],
        help='Group size when quantizing activations. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--reorder', action='store_true',
        help='Whether to keep salient weight unquantized.'
    )
    parser.add_argument(
        '--act_sort_metric', type=str, default='hessian', choices=['abs_mean', 'hessian'],
        help='The metric used to sort the activations.'
    )
    parser.add_argument(
        '--keeper', type=int, default=0,
        help='Group size to keep outliers.'
    )
    parser.add_argument(
        '--keeper_precision', type=int, default=0, choices=[0, 1, 2, 3],
        help='Precision to keep outliers. 0 for FP16; 1 for E5M2; 2 for E4M3; 3 for INT8 Quant.'
    )
    parser.add_argument(
        '--cache_index', action='store_true',
        help='Whether to use cached reorder index'
    )
    parser.add_argument(
        '--tiling', type=int, default=0, choices=[0, 16],
        help='Tile-wise quantization granularity (Deprecated in Atom).'
    )
    parser.add_argument(
        '--kv_cache', action='store_true',
        help='Whether to quant KV_Cache'
    )
    parser.add_argument(
        '--use_gptq', action='store_true',
        help='Whether to use GPTQ for weight quantization.'
    )
    parser.add_argument(
        '--smoothquant', action='store_true',
        help='Whether to use SmoothQuant.'
    )
    parser.add_argument(
        '--static_scales', type=str, default=None,
        help='Path to the staitc activation scales for SmoothQuant.'
    )
    parser.add_argument(
        '--smooth_scales', type=str, default=None,
        help='Path to the smooth scales for SmoothQuant.'
    )
    parser.add_argument(
        '--quantize_output', action='store_true',
        help='Whether to quantize the output of INT8 Matrix Multiplication.'
    )
    parser.add_argument(
        '--skip_down_proj', action='store_true',
        help='Whether to skip quantizing down_proj in LlamaMLP.'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Alpha value for SmoothQuant.'
    )
    parser.add_argument(
        '--w_quant', type=str, default='per_tensor', choices=['per_tensor', 'per_channel', 'per_8_channel'],
        help='Type of weight quantization for SmoothQuant.'
    )
    parser.add_argument(
        '--a_quant', type=str, default='per_tensor', choices=['per_tensor', 'per_token'],
        help='Type of activation quantization for SmoothQuant.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--w_clip_ratio', type=float, default=1.0,
        help='Clip ratio for weight quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--kv_clip_ratio', type=float, default=1.0,
        help='Clip ratio for kv cache quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        "--eval_ppl", action="store_true",
        help='Whether to evaluate perplexity.'
    )
    parser.add_argument(
        "--eval_common_sense", action="store_true",
        help='Whether to evaluate zero-shot accuray on commonsense reasoning tasks.'
    )
    parser.add_argument(
        "--multigpu", action="store_true", 
        help="at eval, map model to multiple gpus"
    )
    parser.add_argument(
        "--lm_eval_num_fewshot", type=int, default=0, 
        help="Number of shots in lm evaluation. Default is 0 for zero-shot."
    )
    parser.add_argument(
        "--lm_eval_limit", type=int, default=-1, 
        help="Limit the number of examples in lm evaluation"
    )
    parser.add_argument(
        '--save_dir', type=str, default='./saved',
        help='Path to store the reordering indices and quantized weights.'
    )
    parser.add_argument(
        '--save_model', type=str, default=None,
        help='Path to store the quantized model.'
    )
    
    # microxscaling settings
    parser.add_argument("--mx", action="store_true", help="Whether to use microxcaling")
    parser.add_argument("--mx_format", type=str, choices=["int8", "int4", "fp8_e5m2", "fp8_e4m3",
                            "fp6_e3m2", "fp6_e2m3", "fp4_e2m1"], help="MX element format")
    parser.add_argument("--mx_block_size", type=int, default=32, help="MX block size")
    
    args = parser.parse_args()

    model_name = args.model.lower().split('/')[-1]
    assert model_name != None, "Please check the model path."

    if "llama" in args.model.lower():
        model = get_llama(args)
        get_act_stats_func = get_act_stats_llama
        reorder_model_func = reorder_model_llama
        add_act_quant_wrapper_func = add_act_quant_wrapper_llama
        quantize_model_gptq_func = quantize_model_gptq_llama
        quantize_model_func = quantize_model_llama
        eval_func = llama_eval
    elif "opt" in args.model.lower():
        model = get_opt(args)
        get_act_stats_func = get_act_stats_opt
        reorder_model_func = reorder_model_opt
        add_act_quant_wrapper_func = add_act_quant_wrapper_opt
        quantize_model_gptq_func = quantize_model_gptq_opt
        quantize_model_func = quantize_model_opt
        eval_func = opt_eval
    model.eval()

    if args.reorder:
        if args.cache_index == False:
            dataloader, testloader = get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print("Getting activation stats...")
            act_scales = get_act_stats_func(
                model, dataloader, DEV, metric=args.act_sort_metric
            )
            
            print("Getting reording index...")
            reorder_index = get_reorder_index(model, act_scales)

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(reorder_index, f'{args.save_dir}/{model_name}_reorder_index_{args.dataset}.pt')
        else:
            index_filename = f'{args.save_dir}/{model_name}_reorder_index_{args.dataset}.pt'
            assert os.path.isfile(index_filename), "reorder index file not found."

            print("Loading cached reording index from disk...")
            reorder_index = torch.load(index_filename)

        print("Reordering model...")
        model = reorder_model_func(
            model, device=DEV, args=args, reorder_index=reorder_index
        )
    
    if args.abits < 16:
        print("Inserting activations quantizers ...")
        scales = defaultdict(lambda: None)
        model = add_act_quant_wrapper_func(model, device=DEV, args=args, scales=scales)

    if args.wbits < 16:
        print("Quantizing...")
        if args.use_gptq:
            dataloader, testloader = get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            model = quantize_model_gptq_func(model, device=DEV, args=args, dataloader=dataloader)
        else:
            model = quantize_model_func(model, device=DEV, args=args)

    if args.smooth_scales:
        print("Smooth...")
        from smoothquant.smooth import smooth_lm
        smooth_scales = torch.load(args.smooth_scales)
        smooth_lm(model, smooth_scales, args.alpha)

    if args.smoothquant:
        assert args.abits == 16 and args.wbits == 16, "SmoothQuant only works without Atom."
        print("SmoothQuant...")
        from smoothquant.quant import quantize_llama
        if args.static_scales:
            print("Static scales provided. Using static scales for SmoothQuant.")
        scales = torch.load(args.static_scales) if args.static_scales else None
        model = quantize_llama(model, weight_quant=args.w_quant, act_quant=args.a_quant,
                               scales=scales, a_clip_ratio=args.a_clip_ratio, w_clip_ratio=args.w_clip_ratio,
                               quantize_output=args.quantize_output, skip_down_proj=args.skip_down_proj)

    if (args.mx):
        print("Using microxscaling dataformat.")
        from mx import mx_mapping
        from mx import finalize_mx_specs
        mx_specs = {
        'w_elem_format': args.mx_format, #'int8',#'fp6_e3m2',
        'a_elem_format': args.mx_format, #'int8',#'fp6_e3m2',
        'block_size': args.mx_block_size, #32,
        'bfloat': 16,
        'custom_cuda': True,
        # For quantization-aware finetuning, do backward pass in FP32
        'quantize_backprop': False,
        }
        mx_specs = finalize_mx_specs(mx_specs)
        mx_mapping.inject_pyt_ops(mx_specs)

    if args.save_model:
        torch.save(model, args.save_model)

    if args.eval_ppl:
        datasets = ['wikitext2']
        # datasets = ['wikitext2', 'ptb', 'c4']

        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(f"Evaluating {dataset} ...")
            ppl = eval_func(model, testloader, DEV)

            print(f"targetResult,{dataset},{ppl:.3f}")
    
    # eval zero shot accuracy on commonsense datasets
    if args.eval_common_sense:
        lm = LMClass(args, model)
        lm.seqlen = 2048
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False

        if args.multigpu:
            if "llama" in args.model.lower():
                map_layers_to_multi_gpus(lm.model.model.layers)
                input_device = lm.model.model.layers[0].device
                output_device = lm.model.model.layers[-1].device
                assert input_device == output_device
                lm._device = input_device
                lm.model.model.embed_tokens.to(input_device)
                lm.model.model.norm.to(output_device)
                lm.model.lm_head.to(output_device)
            elif "opt" in args.model.lower():
                map_layers_to_multi_gpus(lm.model.model.decoder.layers)
                input_device = lm.model.model.decoder.layers[0].device
                output_device = lm.model.model.decoder.layers[-1].device
                assert input_device == output_device
                lm._device = input_device
                lm.model.model.decoder.embed_tokens.to(input_device)
                lm.model.model.decoder.embed_positions.to(input_device)
                lm.model.model.decoder.final_layer_norm.to(input_device)
                lm.model.lm_head.to(output_device)
        else:
            lm._device = DEV
            lm.model = lm.model.to(lm.device)

        results = {}
        tasks_str = "piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande"
        task_names = pattern_match(tasks_str.split(","), lm_tasks.ALL_TASKS)
        print(f"Selected Tasks: {task_names}")

        task_dict = lm_tasks.get_task_dict(task_names)
        t_results = lm_evaluator.evaluate(
            lm,
            task_dict,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=None if args.lm_eval_limit == -1 else args.lm_eval_limit
        )
        results.update(t_results)
        pprint(results)

        results_dict = results['results']
        for task_name in tasks_str.split(','):
            if task_name in ['piqa', 'arc_easy', 'arc_challenge', 'hellaswag']:
                print(f"INFO {task_name} : {results_dict[task_name]['acc_norm']*100:.2f}")
            else:
                print(f"INFO {task_name} : {results_dict[task_name]['acc']*100:.2f}")
