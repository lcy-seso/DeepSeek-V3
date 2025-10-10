import pytest
import torch
import re

import json
from model import ModelArgs

from model import Transformer as DeepSeekV3Transformer
from tilert.models.deepseek_v3.model import (
    Transformer as TilertDeepSeekV3Transformer,
)
from safetensors.torch import load_file
import os
from tqdm import tqdm, trange
from glob import glob
from safetensors.torch import safe_open
import torch

@pytest.fixture(autouse=True)
def setup():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)

# output state_dict对齐target state_dict，
# 如果按照正则匹配，匹配完后key 在target state_dict中，则1.将origin state_dict key的weight赋值给对应 weight
# 如果正则不匹配，又在target state_dict中，则将origin state_dict的value赋值给target state_dict
def convert_state_dict_origin2tilert(origin_state_dict: dict, target_state_dict: dict, layer_index:int) -> dict:
    """
    Convert the state dict of the Tilert DeepSeekV3 model to the state dict of the DeepSeekV3 model.
    """
    
    # key_map 是origin to modified namespace 的映射
    key_map = {
        # RMSNorm weight
        r"^layers\.0\.ffn_norm\.weight$": "layers.0.ffn.norm_up_gate.norm_weight",
        r"^layers\.1\.ffn_norm\.weight$": "layers.1.ffn.rmsnorm_expert.norm_weight",

        # MLP weight
        r"layers\.(\d+)\.ffn\.w1\.weight": r"layers.\1.ffn.norm_up_gate.w1.weight",
        r"layers\.(\d+)\.ffn\.w3\.weight": r"layers.\1.ffn.norm_up_gate.w3.weight",

        r"layers\.(\d+)\.ffn\.w1\.scale": r"layers.\1.ffn.norm_up_gate.w1.scale",
        r"layers\.(\d+)\.ffn\.w3\.scale": r"layers.\1.ffn.norm_up_gate.w3.scale",

        r"layers\.(\d+)\.ffn\.w2\.weight": r"layers.\1.ffn.down.w2.weight",
        r"layers\.(\d+)\.ffn\.w2\.scale": r"layers.\1.ffn.down.w2.scale",

        # Project weight
        r"layers\.(\d+)\.ffn\.gate\.weight": r"layers.\1.ffn.rmsnorm_expert.proj_weight",        
        r"layers\.(\d+)\.ffn\.gate\.bias": r"layers.\1.ffn.routed_up_gate_silu.route_gate.bias",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.weight": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.experts_w1.\2.weight",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.scale": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.experts_w1.\2.scale",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w3\.weight": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.experts_w3.\2.weight",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w3\.scale": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.experts_w3.\2.scale",
        r"layers\.(\d+)\.ffn\.shared_experts\.w1\.weight": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w1.weight",
        r"layers\.(\d+)\.ffn\.shared_experts\.w1\.scale": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w1.scale",
        r"layers\.(\d+)\.ffn\.shared_experts\.w3\.weight": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w3.weight",
        r"layers\.(\d+)\.ffn\.shared_experts\.w3\.scale": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w3.scale",
        
        r"layers\.(\d+)\.ffn\.shared_experts\.w2\.weight": r"layers.\1.ffn.down.shared_experts_w2.weight",
        r"layers\.(\d+)\.ffn\.shared_experts\.w2\.scale": r"layers.\1.ffn.down.shared_experts_w2.scale",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w2\.weight": r"layers.\1.ffn.down.experts_w2.\2.weight",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w2\.scale": r"layers.\1.ffn.down.experts_w2.\2.scale",
        # norm attn
        r"layers\.(\d+)\.attn\.wq_a\.weight": r"layers.\1.attn.rmsnorm_proj_qkvwa_rope.wq_a.weight",
        r"layers\.(\d+)\.attn\.wq_a\.scale": r"layers.\1.attn.rmsnorm_proj_qkvwa_rope.wq_a.scale",
        r"layers\.(\d+)\.attn\.wkv_a\.weight": r"layers.\1.attn.rmsnorm_proj_qkvwa_rope.wkv_a.weight",
        r"layers\.(\d+)\.attn\.wkv_a\.scale": r"layers.\1.attn.rmsnorm_proj_qkvwa_rope.wkv_a.scale",
        r"layers\.(\d+)\.attn_norm\.weight": r"layers.\1.attn.rmsnorm_proj_qkvwa_rope.attn_norm.weight",
        # op2
        r"layers\.(\d+)\.attn\.wq_b\.weight": r"layers.\1.attn.rmsnorm_proj_qwb_rope.wq_b.weight",
        r"layers\.(\d+)\.attn\.wq_b\.scale": r"layers.\1.attn.rmsnorm_proj_qwb_rope.wq_b.scale",
        r"layers\.(\d+)\.attn\.q_norm\.weight": r"layers.\1.attn.rmsnorm_proj_qwb_rope.q_norm.weight",
        # op3
        r"layers\.(\d+)\.attn\.wkv_b\.weight": r"layers.\1.attn.proj_qwb.wkv_b.weight",
        r"layers\.(\d+)\.attn\.wkv_b\.scale": r"layers.\1.attn.proj_qwb.wkv_b.scale",
        # op4
        r"layers\.(\d+)\.attn\.kv_norm\.weight": r"layers.\1.attn.kv_rmsnorm.kv_norm_weight",
        # op7
        r"layers\.(\d+)\.attn\.wo\.weight": r"layers.\1.attn.unproj_o_allreduce.wo.weight",
        r"layers\.(\d+)\.attn\.wo\.scale": r"layers.\1.attn.unproj_o_allreduce.wo.scale",
    }
    key_map_origin2tilert = {
        # mla

        r"layers\.(\d+)\.attn_norm\.weight": r"layers.\1.attn.rmsnorm_proj_qkvwa_rope.attn_norm.weight",
        r"layers\.(\d+)\.attn\.wq_a\.weight": r"layers.\1.attn.rmsnorm_proj_qkvwa_rope.wq_a.weight",
        r"layers\.(\d+)\.attn\.wq_a\.scale": r"layers.\1.attn.rmsnorm_proj_qkvwa_rope.wq_a.scale",
        r"layers\.(\d+)\.attn\.q_norm\.weight": r"layers.\1.attn.rmsnorm_proj_qwb_rope.q_norm.weight", 
        r"layers\.(\d+)\.attn\.wq_b\.weight": r"layers.\1.attn.rmsnorm_proj_qwb_rope.wq_b.weight",
        r"layers\.(\d+)\.attn\.wq_b\.scale": r"layers.\1.attn.rmsnorm_proj_qwb_rope.wq_b.scale",
        r"layers\.(\d+)\.attn\.wkv_a\.weight": r"layers.\1.attn.rmsnorm_proj_qkvwa_rope.wkv_a.weight",
        r"layers\.(\d+)\.attn\.wkv_a\.scale": r"layers.\1.attn.rmsnorm_proj_qkvwa_rope.wkv_a.scale",
        r"layers\.(\d+)\.attn\.kv_norm\.weight": r"layers.\1.attn.kv_rmsnorm.kv_norm.weight",
        r"layers\.(\d+)\.attn\.wkv_b\.weight": r"layers.\1.attn.proj_qwb.wkv_b.weight",
        r"layers\.(\d+)\.attn\.wkv_b\.scale": r"layers.\1.attn.proj_qwb.wkv_b.scale",
        r"layers\.(\d+)\.attn\.wo\.weight": r"layers.\1.attn.unproj_o_allreduce.wo.weight",
        r"layers\.(\d+)\.attn\.wo\.scale": r"layers.\1.attn.unproj_o_allreduce.wo.scale",
        # mlp
        # r"^layers\.(\d+)\.ffn_norm\.weight": r"layers.\1.ffn.norm_up_gate.norm_weight", # 会被覆盖
        r"layers\.(\d+)\.ffn\.w1\.weight": r"layers.\1.ffn.norm_up_gate.w1.weight",
        r"layers\.(\d+)\.ffn\.w1\.scale": r"layers.\1.ffn.norm_up_gate.w1.scale",
        r"layers\.(\d+)\.ffn\.w2\.weight": r"layers.\1.ffn.down.w2.weight",
        r"layers\.(\d+)\.ffn\.w2\.scale": r"layers.\1.ffn.down.w2.scale",
        r"layers\.(\d+)\.ffn\.w3\.weight": r"layers.\1.ffn.norm_up_gate.w3.weight",
        r"layers\.(\d+)\.ffn\.w3\.scale": r"layers.\1.ffn.norm_up_gate.w3.scale",
        # moe
        # deepseek to tilert的时候会一变多
        r"layers\.(\d+)\.ffn_norm\.weight": [r"layers.\1.ffn.rmsnorm_expert.norm_weight", r"layers.\1.ffn.norm_up_gate.norm_weight"], # 会覆盖前面的
        r"layers\.(\d+)\.ffn\.gate\.weight": r"layers.\1.ffn.rmsnorm_expert.proj_weight",
        # r"layers\.(\d+)\.ffn\.gate\.route_scale": r"layers.\1.ffn.rmsnorm_expert.weight.scale", # ?
        r"layers\.(\d+)\.ffn\.gate\.bias": r"layers.\1.ffn.routed_up_gate_silu.route_gate.bias",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.weight": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.experts_w1.\2.weight",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.scale": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.experts_w1.\2.scale",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w3\.weight": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.experts_w3.\2.weight",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w3\.scale": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.experts_w3.\2.scale",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w2\.weight": r"layers.\1.ffn.down.experts_w2.\2.weight",
        r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w2\.scale": r"layers.\1.ffn.down.experts_w2.\2.scale",
        r"layers\.(\d+)\.ffn\.shared_experts\.w1\.weight": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w1.weight",
        r"layers\.(\d+)\.ffn\.shared_experts\.w1\.scale": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w1.scale",
        r"layers\.(\d+)\.ffn\.shared_experts\.w3\.weight": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w3.weight",
        r"layers\.(\d+)\.ffn\.shared_experts\.w3\.scale": r"layers.\1.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w3.scale",
        r"layers\.(\d+)\.ffn\.shared_experts\.w2\.weight": r"layers.\1.ffn.down.shared_experts_w2.weight",
        r"layers\.(\d+)\.ffn\.shared_experts\.w2\.scale": r"layers.\1.ffn.down.shared_experts_w2.scale",
        # 1569层？算上没有匹配的三层
    }
    set_origin_keys = set(origin_state_dict.keys())
    print(f"origin_state_dict size: {len(set_origin_keys)}")
    
    for k in origin_state_dict.keys():
        with open("origin_state_dict.txt", "a") as f:
            f.write(k + "\n")

    set_target_keys = set()
    set_target_layer0_keys = set()
    set_target_layer1_keys = set()
    for k in target_state_dict.keys():
        k = re.sub(r"layers\.1", rf"layers.{layer_index}", k)
        set_target_keys.add(k)
    #     if k.startswith("layers.1"):
    #         set_target_layer1_keys.add(k)
    #     if k.startswith("layers.0"):
    #         set_target_layer0_keys.add(k)
    #     if "norm_up_gate" in k:
    #         print(k)
    # print(set_target_layer0_keys- (set_target_layer0_keys&set_target_layer1_keys))
    # print(set_target_layer1_keys- (set_target_layer0_keys&set_target_layer1_keys))
    state_dict = {}
    count_match_in = 0
    count_nomatch_in = 0
    for key, value in origin_state_dict.items():
        is_found = False
        for pattern, target_pattern in key_map_origin2tilert.items():
            match = re.match(pattern, key)
            if match:
                if isinstance(target_pattern, list):
                    for target_pattern_ in target_pattern:
                        new_key = re.sub(pattern, target_pattern_, key)
                        if new_key in set_target_keys:
                            state_dict[new_key] = value
                            count_match_in += 1
                            break
                else:
                    new_key = re.sub(pattern, target_pattern, key)
                    if new_key in set_target_keys:
                        state_dict[new_key] = value
                        count_match_in += 1
                        break
                is_found = True
        if not is_found and key in set_target_keys:
            state_dict[key] = value
            count_nomatch_in += 1
    set_exchange_keys = set(state_dict.keys())
    print(set_target_keys - (set_exchange_keys&set_target_keys)) # {'layers.0.ffn.norm_up_gate.norm_weight'},因为一遍多了，按这种映射会覆盖
    print(set_exchange_keys - (set_exchange_keys&set_target_keys))
    print(f"target_state_dict size: {len(set_target_keys)}")
    print(f"count_match_in: {count_match_in}")
    print(f"count_nomatch_in: {count_nomatch_in}")
    assert count_match_in + count_nomatch_in == len(set_target_keys)
    state_dict_return = {}
    for k, v in state_dict.items():
        k = re.sub(rf"layers\.{layer_index}", r"layers.1", k)
        state_dict_return[k] = v
    # for k in state_dict_return.keys():
    #     print(k)
    return state_dict_return

def convert_state_dict_tilert2origin(tilert_state_dict: dict) -> dict:
    """
    Convert the state dict of the Tilert DeepSeekV3 model to the state dict of the DeepSeekV3 model.
    """
    key_casting_maps = {
        # RMSNorm weight
        # 不需要因为order一样
        # r"layers\.(\d+)\.ffn\.norm\.weight": r"layers.\1.ffn_norm.weight",
        r"layers\.(\d+)\.ffn\.norm_up_gate\.norm_weight": r"layers.\1.ffn_norm.weight",  # noqa: E501
        # r"layers\.(\d+)\.ffn\.route_proj\.norm_weight": r"layers.\1.ffn_norm.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.rmsnorm_expert\.norm_weight": r"layers.\1.ffn_norm.weight",  # noqa: E501
        # MLP weight
        r"layers\.(\d+)\.ffn\.norm_up_gate.w1.weight": r"layers.\1.ffn.w1.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.w2.weight": r"layers.\1.ffn.w2.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.norm_up_gate.w3.weight": r"layers.\1.ffn.w3.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.norm_up_gate.w1.scale": r"layers.\1.ffn.w1.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.w2.scale": r"layers.\1.ffn.w2.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.norm_up_gate.w3.scale": r"layers.\1.ffn.w3.scale",  # noqa: E501
        # Project weight
        r"layers\.(\d+)\.ffn\.rmsnorm_expert\.proj_weight": r"layers.\1.ffn.gate.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.route_gate\.bias": r"layers.\1.ffn.gate.bias",  # noqa: E501
        # Expert weight
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w1\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w1.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w1\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w1.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.experts_w2\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w2.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.experts_w2\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w2.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w3\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w3.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w3\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w3.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w1.weight": r"layers.\1.ffn.shared_experts.w1.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w1.scale": r"layers.\1.ffn.shared_experts.w1.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.shared_experts_w2.weight": r"layers.\1.ffn.shared_experts.w2.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.shared_experts_w2.scale": r"layers.\1.ffn.shared_experts.w2.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w3.weight": r"layers.\1.ffn.shared_experts.w3.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w3.scale": r"layers.\1.ffn.shared_experts.w3.scale",  # noqa: E501


        # norm attn
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_rope\.wq_a\.weight": r"layers.\1.attn.wq_a.weight",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_rope\.wq_a\.scale": r"layers.\1.attn.wq_a.scale", 
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_rope\.wkv_a\.weight": r"layers.\1.attn.wkv_a.weight",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_rope\.wkv_a\.scale": r"layers.\1.attn.wkv_a.scale",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_rope\.attn_norm\.weight": r"layers.\1.attn_norm.weight",
        # op2
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qwb_rope\.wq_b\.weight": r"layers.\1.attn.wq_b.weight",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qwb_rope\.wq_b\.scale": r"layers.\1.attn.wq_b.scale",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qwb_rope\.q_norm\.weight": r"layers.\1.attn.q_norm.weight",
        # op3
        r"layers\.(\d+)\.attn\.proj_qwb\.wkv_b\.weight": r"layers.\1.attn.wkv_b.weight",
        r"layers\.(\d+)\.attn\.proj_qwb\.wkv_b\.scale": r"layers.\1.attn.wkv_b.scale", 
         # op4
        r"layers\.(\d+)\.attn\.kv_rmsnorm\.kv_norm\.weight": r"layers.\1.attn.kv_norm.weight",
        # op7
        r"layers\.(\d+)\.attn\.unproj_o_allreduce\.wo\.weight": r"layers.\1.attn.wo.weight",
        r"layers\.(\d+)\.attn\.unproj_o_allreduce\.wo\.scale": r"layers.\1.attn.wo.scale", 
    }

    state_dict = {}
    for key, value in tilert_state_dict.items():
        is_found = False
        for pattern, target_pattern in key_casting_maps.items():
            match = re.match(pattern, key)
            if match:
                new_key = re.sub(pattern, target_pattern, key)
                state_dict[new_key] = value
                is_found = True
                break
        if not is_found:
            state_dict[key] = value
    return state_dict

# 
def state_dict_weight_exchange(total:dict, one_gpu:dict)->dict:
    # 参照one_gpu的shape，切割total切割，分8分，分给8个gpu
    total_device8 = [{} for _ in range(8)]
    for key_total, value_total in total.items():
        value_one_gpu = one_gpu[key_total]
        if value_one_gpu.ndim == 3:
            print(f"value_total shape: {value_total.shape}, value_one_gpu shape: {value_one_gpu.shape}")
            assert("do nothing in dim = 3.")
        # if "wkv_b" in key_total:
        #     if "weight" in key_total:
        #         # (32768, 512)->(2, 16384, 512) ->(2, 8, 2048, 512) ->(8, 2, 2048, 512)->(8, 4096, 512)
        #         value_total = value_total.reshape(2, 16384, 512).reshape(2, 8, 2048, 512).permute(1, 0, 2, 3).reshape(8, 4096, 512)
        #     elif "scale" in key_total:
        #         value_total = value_total.reshape(2, 128, 4).reshape(2, 8, 16, 4).permute(1, 0, 2, 3).reshape(8, 4096, 512)
            
            #     print("wkv_b.scale : ", value_total.shape)
            #     # continue
            
        # 如果value_one_gpu row * 8 = value_total row, 且 value_one_gpu col = value_total col, 那么value_total row 需要切成8份，分给8个gpu
        # 如果value_one_gpu row = value_total row, 且 value_one_gpu col *8 = value_total col, 那么value_total col 需要切成8份，分给8个gpu
        elif value_one_gpu.ndim == 2:
            if value_one_gpu.shape[0] * 8 == value_total.shape[0] and value_one_gpu.shape[1] == value_total.shape[1]:
                # print(f"value_total shape: {value_total.shape}, value_one_gpu shape: {value_one_gpu.shape}")
                value_total = value_total.reshape(8, value_one_gpu.shape[0], -1)
                # print(f"value_total shape: {value_total.shape}, value_one_gpu shape: {value_one_gpu.shape}")
                # print(key_total)
                # break
            elif value_one_gpu.shape[0] == value_total.shape[0] and value_one_gpu.shape[1] * 8 == value_total.shape[1]:
                value_total = value_total.reshape(8, -1, value_one_gpu.shape[1])
                
            elif value_one_gpu.shape[0] == value_total.shape[0] and value_one_gpu.shape[1] == value_total.shape[1]:
                # 复制8份
                value_total = value_total.repeat(8, 1, 1)
            else:
                raise ValueError(f"value_total shape: {value_total.shape}, value_one_gpu shape: {value_one_gpu.shape}")
        
        else:
            # 存在一维的，但是没有8倍关系
            if value_one_gpu.shape[0] == value_total.shape[0]:
                value_total = value_total.repeat(8, 1)
            else:
                raise ValueError(f"value_total shape: {value_total.shape}, value_one_gpu shape: {value_one_gpu.shape}")
        tensors = torch.unbind(value_total, dim=0)
        # 8个dict，每个拿自己的，保存到total_device8
        for i in range(8):
            total_device8[i][key_total] = tensors[i]
        
            
    return total_device8

# to do list: 还需要转换成deepseek_modified key mode!!!!!!!!!!  不需要转换，其实就是同一个。
# load deepseek hf weights，convert to source code key mode, convert to deepseek_modified key mode
def load_origin_state_dicts(model_dir:str, mp:int, n_experts:int)->list:
    # 读取 index 文件
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    mapping = {
        "embed_tokens": ("embed", 0),
        "input_layernorm": ("attn_norm", None),
        "post_attention_layernorm": ("ffn_norm", None),
        "": ("wq", 0),
        "q_a_proj": ("wq_a", None),
        "q_a_layernorm": ("q_norm", None),
        "q_b_proj": ("wq_b", 0),
        "kv_a_proj_with_mqa": ("wkv_a", None),
        "kv_a_layernorm": ("kv_norm", None),
        "kv_b_proj": ("wkv_b", 0),
        "o_proj": ("wo", 1),
        "gate": ("gate", None),
        "gate_proj": ("w1", 0),
        "down_proj": ("w2", 1),
        "up_proj": ("w3", 0),
        "norm": ("norm", None),
        "lm_head": ("head", 0),
        "scale": ("scale", None),
        } 

    state_dicts = [{} for _ in range(mp)]
    n_local_experts = n_experts // mp
    for file_path in tqdm(glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                for i in range(mp):
                    new_param = param
                    # print(new_param)÷
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param
    return state_dicts
    
def compare_state_dicts(sd_a, sd_b, check_values=True, rtol=1e-5, atol=1e-6, max_print=20):
    ok = True
    keys_a, keys_b = set(sd_a.keys()), set(sd_b.keys())
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    if only_a or only_b:
        ok = False
        print(f"Missing in B: {len(only_a)}, Missing in A: {len(only_b)}")
        if only_a: print("  sample A-only:", only_a[:max_print])
        if only_b: print("  sample B-only:", only_b[:max_print])

    common = sorted(keys_a & keys_b)
    shape_mismatch = []
    dtype_mismatch = []
    value_mismatch = []

    for k in common:
        a, b = sd_a[k], sd_b[k]
        if a.shape != b.shape:
            ok = False
            shape_mismatch.append(k)
            if len(shape_mismatch) <= max_print:
                print(f"[shape] {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if a.dtype != b.dtype:
            dtype_mismatch.append(k)
            # 不因 dtype 不同而提前失败，后面转浮点比较
        if check_values and a.numel() > 0:
            # 统一到 CPU + 浮点比较（整数则严格相等）
            if a.is_floating_point() or b.is_floating_point():
                a_f = a.detach().to('cpu', dtype=torch.float32)
                b_f = b.detach().to('cpu', dtype=torch.float32)
                equal = torch.allclose(a_f, b_f, rtol=rtol, atol=atol)
                if not equal:
                    ok = False
                    max_diff = (a_f - b_f).abs().max().item()
                    if len(value_mismatch) < max_print:
                        print(f"[value] {k}: max_diff={max_diff:.6g}, rtol={rtol}, atol={atol}")
                    value_mismatch.append(k)
            else:
                # 非浮点（如 int/uint/bool）严格相等
                equal = torch.equal(a.detach().cpu(), b.detach().cpu())
                if not equal:
                    ok = False
                    if len(value_mismatch) < max_print:
                        print(f"[value] {k}: non-float tensor not equal")
                    value_mismatch.append(k)

    if dtype_mismatch:
        print(f"dtype mismatch: {len(dtype_mismatch)} (sample: {dtype_mismatch[:max_print]})")

    print(f"Summary -> ok={ok}, "
          f"keys_equal={not only_a and not only_b}, "
          f"shape_mismatch={len(shape_mismatch)}, "
          f"dtype_mismatch={len(dtype_mismatch)}, "
          f"value_mismatch={len(value_mismatch)}")
    return ok

@pytest.mark.parametrize("model_config", [])
def test_e2e_forward_pass(model_config):
    with open(model_config, "r", encoding="utf-8") as f_json:
        model_config = json.load(f_json)
    model_args = ModelArgs(**model_config)
    x = torch.randint(0, model_args.vocab_size, (1, 1))
    tilert_model = TilertDeepSeekV3Transformer(model_args)
    origin_model = DeepSeekV3Transformer(model_args)

    model_dir = "/data2/shared/deepseekv3.1"
    # deepseek_hf -> deepseek_origin namespace
    state_dicts = load_origin_state_dicts(model_dir, 1, 256)
    # deepseek_origin -> tilert namespace
    # for key, value in state_dicts[0].items():
    #     print(value.shape)
    result = {}
    for layer_index in range(3,4):
        # origin2tilert_state_dict = convert_state_dict_origin2tilert(state_dicts[0], tilert_model.state_dict())
        origin2tilert_state_dict = convert_state_dict_origin2tilert(state_dicts[0], tilert_model.state_dict(), layer_index)
        # 简单统计 tilert_model 的 state_dict 键值对数量
        tilert_state_dict = tilert_model.state_dict()
        print(f'Tilert model state_dict 键值对数量: {len(tilert_state_dict)}')
        print(f'Origin model state_dict 键值对数量: {len(origin_model.state_dict())}')
        print(f'Origin2tilert model state_dict 键值对数量: {len(origin2tilert_state_dict)}')
        
        # compare_state_dicts(origin2tilert_state_dict, tilert_model.state_dict(),check_values=False)
        # shape 转换一下
        for k in tilert_state_dict.keys():
            if "proj_qwb.wkv_b.weight" in k:
                print("proj_qwb.wkv_b.weight", tilert_state_dict[k].shape)
        origin2tilert_state_dict_device8 = state_dict_weight_exchange(origin2tilert_state_dict, tilert_model.state_dict())
        for i in range(8):
            compare_state_dicts(origin2tilert_state_dict_device8[i], tilert_model.state_dict(),check_values=False)
        
        tilert_model.load_state_dict(origin2tilert_state_dict_device8[0])
        origin_model.load_state_dict(convert_state_dict_tilert2origin(tilert_model.state_dict())) 
        # origin_model.load_state_dict(convert_state_dict_tilert2origin(origin2tilert_state_dict_device8[0]))
        ref_output = origin_model(x, start_pos=127)
        tilert_output = tilert_model(x, start_pos=127)
        ref_output = ref_output.to(torch.float32)
        tilert_output = tilert_output.to(torch.float32)
        abs_err = torch.abs(ref_output - tilert_output)
        rel_err = abs_err / torch.abs(ref_output)
        print(f"Rel err: max-{rel_err.max():.3f}/mean-{rel_err.mean():.3f}")
        print(f"Abs err: max-{abs_err.max():.3f}/mean-{abs_err.mean():.3f}")
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_output.flatten(), 
            tilert_output.flatten(), 
            dim=0
        )
        print(f"Cosine similarity: {cos_sim.item():.6f}")
        p1_error = (torch.abs(ref_output.flatten() - tilert_output.flatten()) / (1 + torch.abs(ref_output.flatten()))).to("cpu")
        print(f"P1: max-{p1_error.max()}, min-{p1_error.min()}, mean-{p1_error.mean()}")
        result[layer_index] = [cos_sim.item(), p1_error.max().item(), p1_error.min().item(), p1_error.mean().item()]
    print(result)
def main():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    test_e2e_forward_pass("configs/config_671B_layer2_device1.json")


if __name__ == "__main__":
    main()

# 1.如何验证权重
