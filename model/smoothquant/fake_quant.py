import torch
from torch import nn
from functools import partial


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8, scales=None):
    # w: (out_features, in_features)
    w = w.clone()
    if scales is None:
        scales = w.abs().max(dim=-1, keepdim=True)[0]
    # scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    scales = scales.to(w.device)
    w.div_(scales).round_().clamp_(min=-q_max-1, max=q_max).mul_(scales)
    return w

@torch.no_grad()
def quantize_weight_per_8_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    w = w.clone()
    shape = w.shape
    w = w.reshape(-1, 8, shape[1])
    scales = w.abs().amax(dim=(1, 2), keepdim=True)
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    scales = scales.to(w.device)
    w.div_(scales).round_().clamp_(min=-q_max-1, max=q_max).mul_(scales)
    w = w.reshape(shape)
    return w

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8, scales=None):
    # w: (out_features, in_features)
    w = w.clone()
    scales = scales.max() if scales is not None else w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    scales = scales.to(w.device)
    w.div_(scales).round_().clamp_(min=-q_max-1, max=q_max).mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, scales=None):
    # static_scales = scales.copy()
    # scales = None
    # if scales is not None:
    #     print(f"Dynamic scales: {t.abs().max(dim=-1, keepdim=True)[0]}")
    #     print(f"Static scales: {scales}, max: {scales.max()}")
    if scales is None:
        scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5).div(q_max).to(t.device)
    t.div_(scales).round_().clamp_(min=-q_max-1, max=q_max).mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, scales=None):
    # if scales is not None and scales.max() > 100:
    #     return t
    # static_scales = scales.clone().max()
    # scales = None
    # if scales is not None:
    #     print(f"Dynamic scales: {t.abs().max()}, Static scales: {scales.max()}")
    scales = scales.max() if scales is not None else t.abs().max()
    # scales = scales.clamp(max=50.0)
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5).div(q_max).to(t.device)
    t.div_(scales).round_().clamp_(min=-q_max-1, max=q_max).mul_(scales)
    return t


class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        scales=None,
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
        
        
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=8, 
                scales=scales['input'] if scales else None
            )
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=8, 
                scales=scales['input'] if scales else None
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = partial(
                self.act_quant.func, n_bits=8, 
                scales=scales['output'] if scales else None
            )
        else:
            self.output_quant_name = "None"
            # self.output_quant = lambda x: x
            self.output_quant = nn.Identity()

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False, scales=None
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            scales=scales,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8, scales=scales['weight'] if scales else None
            )  # use 8-bit integer for weight
        elif weight_quant == "per_8_channel":
            new_module.weight = quantize_weight_per_8_channel_absmax(
                module.weight, n_bits=8
            )
            temp = quantize_weight_per_8_channel_absmax(
                module.weight, n_bits=8
            )
            temp2 = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )
            print(torch.equal(temp, temp2))
            # print(temp)
            # print(temp2)
            
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8, scales=scales['weight'] if scales else None
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_opt(
    model, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=True
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant
            )
            m.fc2 = W8A8Linear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_llama_like(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = W8A8Linear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant
            )
            m.w2 = W8A8Linear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant
            )
            m.w3 = W8A8Linear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = W8A8Linear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_falcon(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = W8A8Linear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant
            )
            m.dense_4h_to_h = W8A8Linear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = W8A8Linear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.dense = W8A8Linear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
