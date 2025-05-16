import torch
from torch import Tensor, device, dtype, nn
from .quantizer import *
from tqdm import tqdm
from bitsandbytes.functional import quantize_4bit, dequantize_4bit

def convertModelToQuant(model, 
                        modules_to_not_convert=["lm_head"], 
                        current_key_name=None, 
                        has_been_replaced=False,
                        compute_dtype=torch.bfloat16, 
                        quant_type="clsq-n2f3", 
                        q_group_size=128):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        if (isinstance(module, nn.Linear)) and name not in modules_to_not_convert:
            #print(name)
            in_features = module.in_features
            out_features = module.out_features
            weight = module.weight
            bias = module.bias
            
            model._modules[name] = QLinear(
                in_features,
                out_features,
                module.bias is not None,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                q_group_size=q_group_size
            )

            model._modules[name].weight = weight
            model._modules[name].bias = bias
            has_been_replaced = True
            # Store the module class in case we need to transpose the weight later
            model._modules[name].source_cls = type(module)
        if len(list(module.children())) > 0:
            _, has_been_replaced = convertModelToQuant(
                module,
                modules_to_not_convert,
                current_key_name,
                has_been_replaced,
                compute_dtype,
                quant_type,
                q_group_size
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced
    
class QLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, compute_dtype=torch.bfloat16, quant_type="ste-n2f3", q_group_size=128, device=None):
        super().__init__(input_features, output_features, bias, device)

        if quant_type == "ste-n2f3":
            self.weight_quantizer = SteN2F3Quantizer(q_group_size=q_group_size)
        elif quant_type == "int2-asym":
            self.weight_quantizer = SteInt2AsymQuantizer(q_group_size=q_group_size)
        elif quant_type == "int3-asym":
            self.weight_quantizer = SteInt3AsymQuantizer(q_group_size=q_group_size)
        elif quant_type == "int4-asym":
            self.weight_quantizer = SteInt4AsymQuantizer(q_group_size=q_group_size)
        elif quant_type == "Q4_0":
            self.weight_quantizer = Q40Quantizer(q_group_size=q_group_size)
        else:
            raise ValueError(f"Has no support {quant_type}. Valid quant_type:[ste-n2f3, int2-asym]")
        # self.quant_type = quant_type
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor):
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        inp_dtype = x.dtype

        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = None

        quantize_weight = self.weight_quantizer(self.weight.to(self.compute_dtype))
        out = F.linear(x, quantize_weight, bias).to(inp_dtype)

        #org_out = F.linear(x, self.weight, bias).to(inp_dtype)
        #err = (out - org_out).pow(2).mean(dim=1)
        #print("loss:", err.mean().item())

        return out


def find_layers(module, layers=[nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def quant_and_dequant_tensor_q4_0(inp, do_transpose=False):
    t = inp.permute(0, 1) if do_transpose else inp
    blocksize = 64
    quant_type = "fp4"
    q, quant_state = quantize_4bit(t, blocksize=blocksize, quant_type=quant_type)
    t = dequantize_4bit(q, quant_state, blocksize=blocksize, quant_type=quant_type)
    t = t.permute(0, 1) if do_transpose else t
    if t.size() != inp.size():
        raise ValueError(
            f"Tensor shape is not euqal. (t {t.size()} and inp {inp.size()})",
        )
    inp.data = t.data


def quant_and_dequant_tensor_q4_0_v2(x, do_transpose=False):
    x = x.permute(0, 1) if do_transpose else x
    # print(x.device)
    org_w_shape = x.shape

    q_group_size = 32
    if q_group_size > 0:
        # print(org_w_shape)
        assert org_w_shape[-1] % q_group_size == 0
        x = x.reshape(-1, q_group_size)
    assert x.dim() == 2

    # Quant.
    x = x.to(dtype=torch.float32, device=x.device)

    abs_max_val = (x.abs().amax(dim=1, keepdim=True)) / -8
    #print("abs_max_val:", abs_max_val)
    x = Floor.apply(torch.minimum(torch.ones_like(x, device=x.device) * 15.0, x * (1.0 / abs_max_val) + 8.5))
    #print("x:", x)

    # Dequant.
    x = (x - 8) * abs_max_val

    x = x.to(dtype=torch.bfloat16, device=x.device)

    x = x.reshape(org_w_shape)

    x = x.permute(0, 1) if do_transpose else x

    return x

def quant_and_dequant_model_q4_0(model):
    layers = model.model.layers
    for i in tqdm(range(0, len(layers)), desc="Quantizing"):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            #print(f"layer {i}, subset {name}")
            weight = None
            from_weights_map = False
            hf_hook = getattr(subset[name], "_hf_hook", None)
            # print('hf_hook',hf_hook)
            if hf_hook is not None:
                weights_map = getattr(hf_hook, "weights_map", None)
                if weights_map is not None:
                    #print("Move weight to cuda")
                    weight = hf_hook.weights_map["weight"].to("cuda")
                    from_weights_map = True
            if weight is None:
                weight = subset[name].weight

            if name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] or \
                name in ["mlp.gate_proj", "mlp.up_proj"]:
                do_transpose = False
            elif name == "self_attn.o_proj" or name == "mlp.down_proj":
                do_transpose = True
            else:
                do_transpose = False

            new_weight = quant_and_dequant_tensor_q4_0_v2(weight, do_transpose)

            if from_weights_map:
                weights_map["weight"] = weight.to("cpu")
            else:
                subset[name].weight.data = new_weight

