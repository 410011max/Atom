from transformers import LlamaForCausalLM
from .configuration_quant_llama import QuantLlamaConfig
import torch
import torch.nn as nn
from functools import partial
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from tqdm import tqdm



@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8, scales=None, clip_ratio=1.0):
    # w: (out_features, in_features)
    w = w.clone()
    scales = scales.max() if scales is not None else w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max).mul_(clip_ratio)
    scales = scales.to(w.device)
    w.div_(scales).round_().clamp_(min=-q_max-1, max=q_max).mul_(scales)
    return w

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, scales=None, clip_ratio=1.0):
    scales = scales.max() if scales is not None else t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5).div(q_max).mul_(clip_ratio).to(t.device)
    t.div_(scales).round_().clamp_(min=-q_max-1, max=q_max).mul_(scales)
    return t

class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_tensor",
        quantize_output=False,
        scales=None,
        w_clip_ratio=1.0,
        a_clip_ratio=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)
        
        
        if act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=8, 
                scales=scales['input'] if scales else None, 
                clip_ratio=a_clip_ratio,
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = partial(
                self.act_quant.func, n_bits=8, 
                scales=scales['output'] if scales else None,
                clip_ratio=a_clip_ratio,
            )
        else:
            self.output_quant_name = "None"
            self.output_quant = nn.Identity()

    # def to(self, *self, **kwself):
    #     super(W8A8Linear, self).to(*self, **kwself)
    #     self.weight = self.weight.to(*self, **kwself)
    #     if self.bias is not None:
    #         self.bias = self.bias.to(*self, **kwself)
    #     return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_tensor", act_quant="per_tensor", quantize_output=False, 
        scales=None, a_clip_ratio=1.0, w_clip_ratio=1.0
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            scales=scales,
            a_clip_ratio=a_clip_ratio,
            w_clip_ratio=w_clip_ratio,
        )
         
        if weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8, scales=scales['weight'] if scales else None,
                clip_ratio=w_clip_ratio,
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"

def quantize_llama(model, weight_quant='per_tensor', act_quant='per_tensor',
                   scales=None, a_clip_ratio=1.0, w_clip_ratio=1.0,
                   quantize_output=False, skip_down_proj=False):
    progress_bar = tqdm(list(model.named_modules()))
    for name, m in tqdm(list(model.named_modules())):
        progress_bar.set_description(f"Processing {name}")
        # if name == "model.layers.0.mlp":
        #     break
        if isinstance(m, LlamaMLP):
            m.gate_proj = W8A8Linear.from_float(m.gate_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_output,
                                                scales=scales[name + ".gate_proj"] if scales else None,
                                                a_clip_ratio=a_clip_ratio, w_clip_ratio=w_clip_ratio)
            m.up_proj = W8A8Linear.from_float(m.up_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_output,
                                              scales=scales[name + ".up_proj"] if scales else None,
                                              a_clip_ratio=a_clip_ratio, w_clip_ratio=w_clip_ratio)
            if not skip_down_proj:
                m.down_proj = W8A8Linear.from_float(m.down_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_output,
                                                    scales=scales[name + ".down_proj"] if scales else None)
        elif isinstance(m, LlamaAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_output,
                                             scales=scales[name + ".q_proj"] if scales else None,
                                             a_clip_ratio=a_clip_ratio, w_clip_ratio=w_clip_ratio)
            m.k_proj = W8A8Linear.from_float(m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_output,
                                             scales=scales[name + ".k_proj"] if scales else None,
                                             a_clip_ratio=a_clip_ratio, w_clip_ratio=w_clip_ratio)
            m.v_proj = W8A8Linear.from_float(m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_output,
                                             scales=scales[name + ".v_proj"] if scales else None,
                                             a_clip_ratio=a_clip_ratio, w_clip_ratio=w_clip_ratio)
            m.o_proj = W8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_output,
                                             scales=scales[name + ".o_proj"] if scales else None,
                                             a_clip_ratio=a_clip_ratio, w_clip_ratio=w_clip_ratio)
    return model

class QuantLlamaForCausalLM(LlamaForCausalLM):
    config_class = QuantLlamaConfig
    def __init__(self, config:QuantLlamaConfig):
        super().__init__(config)
        self.w_quant = config.w_quant
        self.a_quant = config.a_quant
        self.w_clip_ratio = config.w_clip_ratio
        self.a_clip_ratio = config.a_clip_ratio
        self.quantize_output = config.quantize_output
        self.skip_down_proj = config.skip_down_proj
        self.static_scales = config.static_scales
        self.seqlen = 2048

        print("SmoothQuant...")
        if self.static_scales:
            print("Static scales provided. Using static scales for SmoothQuant.")
        scales = torch.load(self.static_scales) if self.static_scales else None
        self = quantize_llama(self, weight_quant=self.w_quant, act_quant=self.a_quant,
                               scales=scales, a_clip_ratio=self.a_clip_ratio, w_clip_ratio=self.w_clip_ratio,
                               quantize_output=self.quantize_output, skip_down_proj=self.skip_down_proj)
                
        