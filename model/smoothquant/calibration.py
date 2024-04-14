import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm


def get_smooth_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    # dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
    # dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def get_static_decoder_layer_scales(
    model,
    tokenizer,
    dataset_path,
    num_samples=512,
    seq_len=512,
):
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]

        # TODO: check the correct static scale
        comming_max_x = x.view(-1, x.shape[-1]).abs().detach().max(dim=-1, keepdim=True)[0]
        if comming_max_x.shape[0] != 2048:
            # fix for the last input which cotaion tokens less than 2048 
            return
        # hidden_dim = x.shape[-1]
        # x = x.view(-1, hidden_dim).abs().detach()
        # comming_max_x = torch.max(x, dim=0)[0].float().cpu()
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = comming_max_x
        else:
            act_dict[name]["input"] = torch.max(
                act_dict[name]["input"], comming_max_x
            )
        if isinstance(y, tuple):
            y = y[0]
        comming_max_y = y.view(-1, y.shape[-1]).abs().detach().max(dim=-1, keepdim=True)[0]
        # hidden_dim = y.shape[-1]
        # y = y.view(-1, hidden_dim).abs().detach()
        # comming_max_y = torch.max(y, dim=0)[0].float().cpu()
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = comming_max_y
        else:
            act_dict[name]["output"] = torch.max(
                act_dict[name]["output"], comming_max_y
            )

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    # dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
    # dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    dataset = dataset.shuffle(seed=42)
    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)
        # mean_scale = np.mean([v["input"] for v in act_dict.values()])
        # pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    # Find weight scales
    for name, m in list(model.named_modules()):
        if isinstance(m, torch.nn.Linear):
            W = m.weight
            scales = W.abs().max(dim=-1, keepdim=True)[0]
            act_dict[name]["weight"] = scales
    
    
    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = defaultdict(dict)
        # self-attention
        scale_dict["attn_input_scale"] = (
            act_dict[f"model.layers.{idx}.self_attn.q_proj"]["input"] / 127
        )
        scale_dict["q_output_scale"] = (
            act_dict[f"model.layers.{idx}.self_attn.q_proj"]["output"] / 127
        )
        scale_dict["k_output_scale"] = (
            act_dict[f"model.layers.{idx}.self_attn.k_proj"]["output"] / 127
        )
        scale_dict["v_output_scale"] = (
            act_dict[f"model.layers.{idx}.self_attn.v_proj"]["output"] / 127
        )
        scale_dict["o_input_scale"] = (
            act_dict[f"model.layers.{idx}.self_attn.o_proj"]["input"] / 127
        )
        # mlp
        scale_dict["mlp_input_scale"] = (
            act_dict[f"model.layers.{idx}.mlp.gate_proj"]["input"] / 127
        )
        scale_dict["gate_output_scale"] = (
            act_dict[f"model.layers.{idx}.mlp.gate_proj"]["output"] / 127
        )
        scale_dict["up_output_scale"] = (
            act_dict[f"model.layers.{idx}.mlp.up_proj"]["output"] / 127
        )
        scale_dict["down_input_scale"] = (
            act_dict[f"model.layers.{idx}.mlp.down_proj"]["input"] / 127
        )
        scale_dict["down_output_scale"] = (
            act_dict[f"model.layers.{idx}.mlp.down_proj"]["output"] / 127
        )
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict
