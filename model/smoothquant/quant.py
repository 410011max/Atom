from tqdm import tqdm
from smoothquant.fake_quant import W8A8Linear
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

def quantize_llama(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True, scales=None):
    progress_bar = tqdm(list(model.named_modules()))
    for name, m in tqdm(list(model.named_modules())):
        progress_bar.set_description(f"Processing {name}")
        # if name == "model.layers.1.self_attn":
        #     break
        if isinstance(m, LlamaMLP):
            m.gate_proj = W8A8Linear.from_float(m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,
                                                scales=scales[name + ".gate_proj"] if scales else None)
            m.up_proj = W8A8Linear.from_float(m.up_proj, weight_quant=weight_quant, act_quant=act_quant,
                                              scales=scales[name + ".up_proj"] if scales else None)
            m.down_proj = W8A8Linear.from_float(m.down_proj, weight_quant=weight_quant, act_quant=act_quant,
                                                scales=scales[name + ".down_proj"] if scales else None)
        elif isinstance(m, LlamaAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(m.q_proj, weight_quant=weight_quant, act_quant=act_quant, 
                                             scales=scales[name + ".q_proj"] if scales else None)
            m.k_proj = W8A8Linear.from_float(m.k_proj, weight_quant=weight_quant, act_quant=act_quant, 
                                             scales=scales[name + ".k_proj"] if scales else None)
            m.v_proj = W8A8Linear.from_float(m.v_proj, weight_quant=weight_quant, act_quant=act_quant, 
                                             scales=scales[name + ".v_proj"] if scales else None)
            m.o_proj = W8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant,
                                             scales=scales[name + ".o_proj"] if scales else None)
    return model