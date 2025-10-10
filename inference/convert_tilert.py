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

from tilert.models.deepseek_v3.model import (
    Transformer as TilertDeepSeekV3Transformer,
)

# 加载hf ds3.1，并转换成ds source code namespace
# usage: 
# state_dicts = load_origin_state_dicts("/data2/shared/deepseekv3.1", 1, 256)
def load_origin_state_dicts(model_dir:str,  mp:int, n_experts:int)->list:
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
   
# model_config = "configs/config_671B_layer2_device1.json"
# with open(model_config, "r", encoding="utf-8") as f_json:
#     model_config = json.load(f_json)
# model_args = ModelArgs(**model_config)
# tilert_model = TilertDeepSeekV3Transformer(model_args)
# tilert_state_dict = tilert_model.state_dict()
# print(tilert_state_dict)


def convert_state_dict_origin2tilert(origin_state_dict: dict):
    """
    Convert the state dict of the Tilert DeepSeekV3 model to the state dict of the DeepSeekV3 model.
    """
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
    }
    count_exchange = 0
    count_no_exchange = 0
    state_dict = {}
    set_exchange_keys = set(origin_state_dict.keys())

    for key, value in origin_state_dict.items():
        is_found = False
        for pattern, target_pattern in key_map_origin2tilert.items():
            match = re.match(pattern, key)
            if match:
                # 如果是列表
                if isinstance(target_pattern, list):
                    for target_pattern_ in target_pattern:
                        new_key = re.sub(pattern, target_pattern_, key)
                        state_dict[new_key] = value
                        count_exchange += 1
                            # break
                else:
                    new_key = re.sub(pattern, target_pattern, key)
                    state_dict[new_key] = value
                    count_exchange += 1
                is_found = True
        if not is_found:
            state_dict[key] = value
            count_no_exchange += 1
        
    print("count_exchange : ",count_exchange)
    print("count_no_exchange : ",count_no_exchange) 
    print("origin key len : ", len(set_exchange_keys))
    return state_dict

def RMSNormProjectQKVWaRope_weight_exchange(state_dict: dict):
    # rmsnorm_gamma  torch.Size([7168]) torch.bfloat16
    # wqkv_a  torch.Size([2112, 7168]) torch.float8_e4m3fn
    # wqkv_a_scales torch.Size([129, 64]) torch.bfloat16
    gpus_ = {}
    one_gpu = {}
    for layer_id in range(0,67):
        attn_norm_weight = state_dict.get(f"layers.{layer_id}.attn.rmsnorm_proj_qkvwa_rope.attn_norm.weight") # torch.Size([7168]) torch.bfloat16
        wq_a_weight = state_dict.get(f"layers.{layer_id}.attn.rmsnorm_proj_qkvwa_rope.wq_a.weight") # torch.Size([1536, 7168]) torch.float8_e4m3fn
        wq_a_scale = state_dict.get(f"layers.{layer_id}.attn.rmsnorm_proj_qkvwa_rope.wq_a.scale") # torch.Size([12, 56]) torch.float32
        wkv_a_weight = state_dict.get(f"layers.{layer_id}.attn.rmsnorm_proj_qkvwa_rope.wkv_a.weight") # torch.Size([576, 7168]) torch.float8_e4m3fn
        wkv_a_scale = state_dict.get(f"layers.{layer_id}.attn.rmsnorm_proj_qkvwa_rope.wkv_a.scale") # torch.Size([5, 56]) torch.float32
        if (attn_norm_weight is not None) and (wq_a_weight is not None) and (wq_a_scale is not None)  and (wkv_a_weight is not None)  and (wkv_a_scale is not None):
            # print(attn_norm_weight.shape, attn_norm_weight.dtype)
            # print(wq_a_weight.shape, wq_a_weight.dtype)
            # print(wq_a_scale.shape, wq_a_scale.dtype)
            # print(wkv_a_weight.shape, wkv_a_weight.dtype)
            # print(wkv_a_scale.shape, wkv_a_scale.dtype)
            wqkv_a = torch.cat([wq_a_weight, wkv_a_weight], dim=0)
            wqkv_a_scales_raw = torch.cat([wq_a_scale, wkv_a_scale], dim=0)
            wqkv_a_scales = torch.zeros((17, 64), dtype=torch.bfloat16)
            # swizzle
            for i in range(64):
                wqkv_a_scales[:, i] = wqkv_a_scales_raw[:, ((i % 8) * 8 + i // 8) % 56]
                if ((i % 8) * 8 + i // 8) >= 56:
                    wqkv_a_scales[:, i] = 0.0
            wqkv_a_scales_0 = wqkv_a_scales[:16, :]
            wqkv_a_scales_1 = wqkv_a_scales[16:, :]
            # repeat the scales for qkv so to fit the requirement of our tilert op.
            wqkv_a_scales_0 = wqkv_a_scales_0.reshape((16, 1, 64)).repeat(1, 8, 1).reshape(-1, 64)
            wqkv_a_scales = torch.cat([wqkv_a_scales_0, wqkv_a_scales_1], dim=0)
            assert wqkv_a_scales.shape == (129, 64)
            rmsnorm_gamma = (
                attn_norm_weight.reshape(7168))
            assert rmsnorm_gamma.numel() == 7168
        one_gpu[f"layers.{layer_id}.attn.rmsnorm_proj_qkvwa_rope.rmsnorm_gamma"] = rmsnorm_gamma
        one_gpu[f"layers.{layer_id}.attn.rmsnorm_proj_qkvwa_rope.wqkv_a"] = wqkv_a
        one_gpu[f"layers.{layer_id}.attn.rmsnorm_proj_qkvwa_rope.wqkv_a_scales"] = wqkv_a_scales
    for i in range(8):
        gpus_[i] = one_gpu 
    return gpus_

def RMSNormProjectQWbRoPE_weight_exchange(state_dict: dict):
    # rmsnorm_gamma torch.Size([1536]) torch.bfloat16
    # wq_b torch.Size([3072, 1536]) torch.float8_e4m3fn
    # wq_b_scales_t16 torch.Size([384, 12]) torch.bfloat16
    gpus_ = {i: {} for i in range(8)}
    for layer_id in range(0,67):
        q_norm_weight = state_dict.get(f"layers.{layer_id}.attn.rmsnorm_proj_qwb_rope.q_norm.weight") # torch.Size([1536]) torch.bfloat16
        wq_b_weight = state_dict.get(f"layers.{layer_id}.attn.rmsnorm_proj_qwb_rope.wq_b.weight") # torch.Size([24576, 1536]) torch.float8_e4m3fn
        wq_b_scale = state_dict.get(f"layers.{layer_id}.attn.rmsnorm_proj_qwb_rope.wq_b.scale") # torch.Size([192, 12]) torch.float32
        if (q_norm_weight is not None) and (wq_b_weight is not None) and (wq_b_scale is not None):
            rmsnorm_gamma = (q_norm_weight.reshape(1536))
            assert rmsnorm_gamma.numel() == 1536
            wq_b_weights = wq_b_weight.reshape(8, 3072, 1536)
            wq_b_scales = wq_b_scale.reshape(8, 24, 12)
            
            for i in range(8):
                gpu_wq_b_weight = wq_b_weights[i]
                gpu_wq_b_scale= wq_b_scales[i]
                scale = gpu_wq_b_scale.reshape((24, 1, 12)).repeat(1, 128, 1).reshape(-1, 12)
                scale_t16 = gpu_wq_b_scale.reshape((24, 1, 12)).repeat(1, 16, 1).reshape(-1, 12).to(torch.bfloat16)
                gpus_[i][f"layers.{layer_id}.attn.rmsnorm_proj_qwb_rope.rmsnorm_gamma"]=rmsnorm_gamma
                gpus_[i][f"layers.{layer_id}.attn.rmsnorm_proj_qwb_rope.gpu_wq_b_weight"]=gpu_wq_b_weight
                gpus_[i][f"layers.{layer_id}.attn.rmsnorm_proj_qwb_rope.scale_t16"]=scale_t16
                # print("rmsnorm_gamma.shape", rmsnorm_gamma.shape, rmsnorm_gamma.dtype)
                # print("gpu_wq_b_weight.shape", gpu_wq_b_weight.shape, gpu_wq_b_weight.dtype)
                # print("scale_t16.shape", scale_t16.shape, scale_t16.dtype)
            # print(q_norm_weight.shape, q_norm_weight.dtype)
            # print(wq_b_weight.shape, wq_b_weight.dtype)
            # print(wq_b_scale.shape, wq_b_scale.dtype)   
    return gpus_             

def KVRMSNorm_weight_exchange(state_dict: dict):
    # gamma  torch.Size([512]) torch.bfloat16
    gpus_ = {i: {} for i in range(8)}
    one_gpu = {}
    for layer_id in range(0,67):
        kv_norm_weight = state_dict.get(f"layers.{layer_id}.attn.kv_rmsnorm.kv_norm.weight") # kv_norm_weight torch.Size([512]) torch.bfloat16
        if (kv_norm_weight is not None):
            gamma = kv_norm_weight
            one_gpu[f"layers.{layer_id}.attn.kv_rmsnorm.gamma"] = gamma
            # print("kv_norm_weight",kv_norm_weight.shape, kv_norm_weight.dtype)
    for i in range(8):
        gpus_[i] = one_gpu 
    return gpus_

def ProjectQWb_weight_exchange(state_dict: dict):
    # wkv_b_a:  torch.Size([16, 512, 128]) torch.float8_e4m3fn
    # wkv_b_a_scales:  torch.Size([16, 8, 1]) torch.bfloat16
    # for key, value in state_dict.items():
    #     if "proj_qwb.wkv_b." in key:
    #         print(key)
    gpus_ = {i: {} for i in range(8)}
    for layer_id in range(0,67):
        wkv_b_weight = state_dict.get(f"layers.{layer_id}.attn.proj_qwb.wkv_b.weight") # wkv_b_weight torch.Size([32768, 512]) torch.float8_e4m3fn
        wkv_b_scale = state_dict.get(f"layers.{layer_id}.attn.proj_qwb.wkv_b.scale") # wkv_b_scale torch.Size([256, 4]) torch.float32
        # print()
        if (wkv_b_weight is not None and wkv_b_scale is not None):
            wkv_b_0 = wkv_b_weight.view(128, -1, 512)[:, :128]
            wkv_b_0 = wkv_b_0.view(8, 16, 128, 512)
            wkv_b_0_splited = [wkv_b_0[i].contiguous().view(16, 128, 512) for i in range(8)]
            # transpose each item in the splited
            wkv_b_0_transposed = [wkv_b_0_splited[i].transpose(1, 2).contiguous() for i in range(8)]
            
            wkv_b_scale_0 = wkv_b_scale.to(torch.bfloat16).view(128, -1, 4)[:, :1]
            # print(wkv_b_scale_0.shape)
            wkv_b_scale_0 = wkv_b_scale_0.view(8, 16, 1, 4)
            wkv_b_scale_0_splited = [wkv_b_scale_0[i].contiguous().view(16, 1, 4) for i in range(8)]
            # transpose each item in the splited
            wkv_b_scale_0_transposed = [wkv_b_scale_0_splited[i].transpose(1, 2).contiguous() for i in range(8)]
            wkv_b_scale_0_repeated = [wkv_b_scale_0_transposed[i].view(64, 1, 1).repeat(1, 2, 1).view(16, 8, 1) for i in range(8)]
            for i in range(8):
                gpus_[i][f"layers.{layer_id}.attn.proj_qwb.wkv_b_a"]= wkv_b_0_transposed[i]
                gpus_[i][f"layers.{layer_id}.attn.proj_qwb.wkv_b_a_scales"]= wkv_b_scale_0_repeated[i]
            print("wkv_b_0_transposed",wkv_b_0_transposed[0].shape, wkv_b_0_transposed[0].dtype)
            print("wkv_b_scale_0_repeated",wkv_b_scale_0_repeated[0].shape, wkv_b_scale_0_repeated[0].dtype)
    return gpus_

def Unproj_O_Allreduce_weight_exchange(state_dict: dict):
    # dev_mat_in[dev_id]  torch.Size([7168, 2048]) torch.float8_e4m3fn
    # mat_scale torch.Size([896, 16]) torch.bfloat16
    gpus_ = {i: {} for i in range(8)}
    for layer_id in range(0,67):
        wo_weight = state_dict.get(f"layers.{layer_id}.attn.unproj_o_allreduce.wo.weight") # wo_weight torch.Size([7168, 16384]) torch.float8_e4m3fn
        wo_scale = state_dict.get(f"layers.{layer_id}.attn.unproj_o_allreduce.wo.scale") # wo_scale torch.Size([56, 128]) torch.float32
        if (wo_weight is not None and wo_scale is not None):
            # print("wo_weight",wo_weight.shape, wo_weight.dtype)
            # print("wo_scale",wo_scale.shape, wo_scale.dtype)
            # 沿着第1个维度按大小2048进行分割
            wo_weight_chunks = torch.split(wo_weight, 2048, dim=1)
            wo_weights = torch.stack(wo_weight_chunks, dim=0) # [8, 7168, 2048]
            wo_scale_chunks = torch.split(wo_scale, 16, dim=1)
            wo_scales = torch.stack(wo_scale_chunks, dim=0) # [8, 56, 16]

            for i in range(8):
                gpu_wo_weight = wo_weights[i] # [7168, 2048], torch.float8_e4m3fn
                gpu_scale_raw = wo_scales[i] # [56, 16]
                gpu_scale = gpu_scale_raw.reshape((56, 1, 16)).repeat(1, 16, 1).reshape(896, 16).to(torch.bfloat16) # [896, 16] , torch.float32
                gpus_[i][f"layers.{layer_id}.attn.unproj_o_allreduce.dev_mat_in"]= gpu_wo_weight    
                gpus_[i][f"layers.{layer_id}.attn.unproj_o_allreduce.mat_scale"]= gpu_scale

                # print(f"最终形状: {gpu_wo_weight.shape}, {gpu_wo_weight.dtype}")  # [8, 7168, 2048] 
                # print(f"最终形状: {gpu_scale_raw.shape}")  # [56, 16]
                # print(f"最终形状: {gpu_scale.shape} , {gpu_scale.dtype}")  # [896, 16]) , torch.float32
            # wo.weight.shape  torch.Size([7168, 2048]) torch.float8_e4m3fn
            # wo.scale.shape  torch.Size([56, 16]) torch.float32
    return gpus_

def Unproj_Owb(state_dict: dict):
    # wkv_b_b torch.Size([16, 128, 512]) torch.float8_e4m3fn
    # wkv_b_scales torch.Size([16, 1, 4]) torch.bfloat16
    # for key, value in state_dict.items():
    #     if "proj_qwb." in key:
    #         print(key)
    gpus_ = {i: {} for i in range(8)}
    for layer_id in range(0,67):
        wkv_b_weight = state_dict.get(f"layers.{layer_id}.attn.proj_qwb.wkv_b.weight") # wkv_b_weight torch.Size([32768, 512]) torch.float8_e4m3fn
        wkv_b_scale = state_dict.get(f"layers.{layer_id}.attn.proj_qwb.wkv_b.scale") # wkv_b_scale torch.Size([256, 4]) torch.float32
        # print()
        if (wkv_b_weight is not None and wkv_b_scale is not None):
            print("wkv_b_weight ", wkv_b_weight.shape, wkv_b_weight.dtype)
            print("wkv_b_scale ", wkv_b_scale.shape, wkv_b_scale.dtype)
             
            wkv_b_1 = wkv_b_weight.view(128, -1, 512)[:, -128:]
            # print(wkv_b_1.shape)
            # split to 8 slices in the first dim
            wkv_b_1 = wkv_b_1.view(8, 16, 128, 512)
            wkv_b_1_splited = [wkv_b_1[i].contiguous().view(16, 128, 512) for i in range(8)]
            
            wkv_b_scale_1 = wkv_b_scale.to(torch.bfloat16).view(128, -1, 4)[:, -1:]
            # print(wkv_b_scale_1.shape)
            wkv_b_scale_1 = wkv_b_scale_1.view(8, 16, 1, 4)
            wkv_b_scale_1_splited = [wkv_b_scale_1[i].contiguous().view(16, 1, 4) for i in range(8)]
            for i in range(8):
                gpus_[i][f"layers.{layer_id}.attn.unproj_owb.wkv_b_b"]= wkv_b_1_splited[i]
                gpus_[i][f"layers.{layer_id}.attn.unproj_owb.wkv_b_b_scales"]= wkv_b_scale_1_splited[i]
            # print("wkv_b_1_splited",wkv_b_1_splited[0].shape, wkv_b_1_splited[0].dtype)
            # print("wkv_b_scale_1_splited",wkv_b_scale_1_splited[0].shape, wkv_b_scale_1_splited[0].dtype)
    return gpus_

def RMSNormMLPUpGateSiLU_weight_exchange(state_dict: dict):
    # weights  torch.Size([9, 512, 7168]) torch.float8_e4m3fn
    # scales_swizzled  torch.Size([9, 4, 64]) torch.bfloat16
    # for key, value in state_dict.items():
    #     if "norm_up_gate" in key:
    #         print(key)
    gpus_ = {i: {} for i in range(8)}
    for layer_id in range(0,67):
        norm_weight = state_dict.get(f"layers.{layer_id}.ffn.norm_up_gate.norm_weight") # norm_weight torch.Size([7168]) torch.bfloat16
        w1_weight = state_dict.get(f"layers.{layer_id}.ffn.norm_up_gate.w1.weight") # w1_weight torch.Size([18432, 7168]) torch.float8_e4m3fn
        w1_scale = state_dict.get(f"layers.{layer_id}.ffn.norm_up_gate.w1.scale") # w1_scale torch.Size([144, 56]) torch.float32
        w3_weight = state_dict.get(f"layers.{layer_id}.ffn.norm_up_gate.w3.weight") # w3_weight torch.Size([18432, 7168]) torch.float8_e4m3fn
        w3_scale = state_dict.get(f"layers.{layer_id}.ffn.norm_up_gate.w3.scale") # w3_scale torch.Size([144, 56]) torch.float32
        # print(w1_weight)
        if (norm_weight is not None and w1_weight is not None and w1_scale is not None and w3_weight is not None and w3_scale is not None):

            # norm_weight,  torch.Size([7168]) torch.bfloat16
            # w1.weight,  torch.Size([2304, 7168]) torch.float8_e4m3fn
            # w1.scale,  torch.Size([18, 56]) torch.float32
            # w3.weight,  torch.Size([2304, 7168]) torch.float8_e4m3fn
            # w3.scale,  torch.Size([18, 56]) torch.float32
            gpu_w1_weights = torch.stack(torch.chunk(w1_weight, 8, dim=0), dim=0) # gpu_w1_weights torch.Size([8, 2304, 7168]) torch.float8_e4m3fn
            gpu_w1_scales = torch.stack(torch.chunk(w1_scale, 8, dim=0), dim=0)   # gpu_w1_scales torch.Size([8, 18, 56]) torch.float32
            gpu_w3_weights = torch.stack(torch.chunk(w3_weight, 8, dim=0), dim=0) # gpu_w3_weights torch.Size([8, 2304, 7168]) torch.float8_e4m3fn
            gpu_w3_scales = torch.stack(torch.chunk(w3_scale, 8, dim=0), dim=0)   # gpu_w3_scales torch.Size([8, 18, 56]) torch.float32
            for i in range(8):
                w1_weights = gpu_w1_weights[i][None, :, :]
                w1_scales = gpu_w1_scales[i][None, :, :]
                w3_weights = gpu_w3_weights[i][None, :, :]
                w3_scales = gpu_w3_scales[i][None, :, :]
                # Preprocessing: interleave w1 and w3
                # so that each block processes 4x7168 weights for each matrix
                w1_weights = w1_weights.reshape(9, 128, 2, 7168)
                w3_weights = w3_weights.reshape(9, 128, 2, 7168)
                weights = torch.cat([w1_weights, w3_weights], dim=2).reshape(9, 512, 7168)
                # also interleave scales, each 64 blocks share 2x56 scales, for w1/w3 respectively
                w1_scales = w1_scales.reshape(9, 2, 1, 56)
                w3_scales = w3_scales.reshape(9, 2, 1, 56)
                scales = torch.cat([w1_scales, w3_scales], dim=2).reshape(9, 4, 56)
                # Do swizzle for the -1 dim
                scales_swizzled = torch.zeros(9, 4, 64)
                for j in range(64):
                    scales_swizzled[..., j] = scales[..., ((j % 8) * 8 + j // 8) % 56]
                    if ((j % 8) * 8 + j // 8) >= 56:
                        scales_swizzled[..., j] = 0.0

                # print("weights",weights.shape, weights.dtype)
                # print("scales_swizzled",scales_swizzled.shape, scales_swizzled.dtype)
                gpus_[i][f"layers.{layer_id}.ffn.norm_up_gate.weights"]= weights 
                gpus_[i][f"layers.{layer_id}.ffn.norm_up_gate.scales_swizzled"]= scales_swizzled 
            # print(f"{layer_id}")    # 0,1,2
            # print("norm_weight",norm_weight.shape, norm_weight.dtype)
            # print("gpu_w1_weights",gpu_w1_weights.shape, gpu_w1_weights.dtype)
            # print("gpu_w1_scales",gpu_w1_scales.shape, gpu_w1_scales.dtype)
            # print("gpu_w3_weights",gpu_w3_weights.shape, gpu_w3_weights.dtype)
            # print("gpu_w3_scales",gpu_w3_scales.shape, gpu_w3_scales.dtype)
    return gpus_

# mlp down
def Down_Allreduce_weight_exchange(state_dict: dict):
    # for key, value in state_dict.items():
    #     if ".down.w2" in key:
    #         print(key)
    gpus_ = {i: {} for i in range(8)}
    for layer_id in range(0,67):
        w2_weight = state_dict.get(f"layers.{layer_id}.ffn.down.w2.weight") # w2_weight torch.Size([7168, 18432]) torch.float8_e4m3fn
        w2_scale = state_dict.get(f"layers.{layer_id}.ffn.down.w2.scale") # w2_scale torch.Size([56, 144]) torch.float32
        if (w2_weight is not None and w2_scale is not None):
            # print("w2_weight",w2_weight.shape, w2_weight.dtype)
            # print("w2_scale",w2_scale.shape, w2_scale.dtype)
            gpu_w2_weights = torch.stack(torch.split(w2_weight, 2304, dim=1), dim=0)
            gpu_w2_scales = torch.stack(torch.split(w2_scale, 18, dim=1), dim=0)
            for i in range(8):
                gpu_w2_weight = torch.stack(torch.split(gpu_w2_weights[i], 256, dim=1), dim=0) 
                gpu_w2_scale = torch.stack(torch.split(gpu_w2_scales[i], 2,  dim=1), dim=0)
                mat_scale_tilert = gpu_w2_scale
                mat_scale_tilert = (
                    mat_scale_tilert.reshape(9, 56, 1, 2).repeat(1, 1, 16, 1).reshape(9, 128, 14)
                )
                padding_zeros = torch.zeros((9, 128, 2), dtype=torch.bfloat16, device=gpu_w2_weight.device)
                mat_scale_tilert = torch.cat([mat_scale_tilert, padding_zeros], dim=2)
                mat_scale_tilert = mat_scale_tilert.reshape(9, 1024, 2).to(torch.bfloat16)
                gpus_[i][f"layers.{layer_id}.ffn.down.gpu_w2_weight"]= gpu_w2_weight 
                gpus_[i][f"layers.{layer_id}.ffn.down.mat_scale_tilert"]= mat_scale_tilert 

                # print("gpu_w2_weight",gpu_w2_weight.shape, gpu_w2_weight.dtype)
                # print("gpu_w2_scale",gpu_w2_scale.shape, gpu_w2_scale.dtype)
                # print("mat_scale_tilert",mat_scale_tilert.shape, mat_scale_tilert.dtype)
            # dev_mat_in[dev_id] torch.Size([9, 7168, 256]) torch.float8_e4m3fn
            # dev_mat_scale[dev_id] torch.Size([9, 56, 2]) torch.bfloat16
            # mat_scale_tilert torch.Size([9, 1024, 2]) torch.bfloat16
    return gpus_

def RMSNormExpertProj_weight_exchange(state_dict: dict):
    # norm_weight torch.Size([7168]) torch.bfloat16
    # proj_weight torch.Size([256, 7168]) torch.bfloat16
    gpus_ = {i: {} for i in range(8)}
    one_gpu = {}
    for layer_id in range(0,67):
        norm_weight = state_dict.get(f"layers.{layer_id}.ffn.rmsnorm_expert.norm_weight") # norm_weight torch.Size([7168]) torch.bfloat16
        proj_weight = state_dict.get(f"layers.{layer_id}.ffn.rmsnorm_expert.proj_weight") # proj_weight torch.Size([256, 7168]) torch.bfloat16
        if (norm_weight is not None and proj_weight is not None):
            
            print("norm_weight",norm_weight.shape, norm_weight.dtype)
            print("proj_weight",proj_weight.shape, proj_weight.dtype) 
            one_gpu[f"layers.{layer_id}.ffn.rmsnorm_expert.norm_weight"] = norm_weight
            one_gpu[f"layers.{layer_id}.ffn.rmsnorm_expert.proj_weight"] = proj_weight
    for i in range(8):
        gpus_[i] = one_gpu 
    return gpus_

def MoeRoutedUpGateSilu_weight_exchange(state_dict: dict):
    gpus_ = {i: {} for i in range(8)}
    one_gpu = {}
    for layer_id in range(0,67):
        # bias  torch.Size([256]) torch.float32
        # shared_experts_w1_weight torch.Size([2048, 7168]) torch.float8_e4m3fn
        # shared_experts_w1_scale torch.Size([16, 56]) torch.float32
        # shared_experts_w3_weight torch.Size([2048, 7168]) torch.float8_e4m3fn
        # shared_experts_w3_scale torch.Size([16, 56]) torch.float32
        bias = state_dict.get(f"layers.{layer_id}.ffn.routed_up_gate_silu.route_gate.bias")
        shared_experts_w1_weight =  state_dict.get(f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w1.weight") 
        shared_experts_w1_scale =  state_dict.get(f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w1.scale") 
        shared_experts_w3_weight =  state_dict.get(f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w3.weight") 
        shared_experts_w3_scale =  state_dict.get(f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.shared_experts_w3.scale") 
        if (bias is not None and shared_experts_w1_weight is not None and shared_experts_w1_scale is not None and shared_experts_w3_weight is not None and shared_experts_w3_scale is not None):
            expert_w1_weights = [shared_experts_w1_weight.view(1, 2048, 7168)]
            expert_w3_weights = [shared_experts_w3_weight.view(1, 2048, 7168)]
            expert_w1_scales = [shared_experts_w1_scale.view(1, 16, 56)]
            expert_w3_scales = [shared_experts_w3_scale.view(1, 16, 56)]
            # print("shared_experts_w1_scale", shared_experts_w1_scale.shape)
            for expert_index in range(256):
                # expert_w1_weight torch.Size([2048, 7168]) torch.float8_e4m3fn
                expert_w1_weight = state_dict.get(f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.experts_w1.{expert_index}.weight") 
                expert_w1_weights.append(expert_w1_weight.view(1, 2048, 7168)) 
                expert_w3_weight = state_dict.get(f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.experts_w3.{expert_index}.weight") 
                expert_w3_weights.append(expert_w3_weight.view(1, 2048, 7168)) 
                expert_w1_scale = state_dict.get(f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.experts_w1.{expert_index}.scale") 
                expert_w1_scales.append(expert_w1_scale.view(1, 16, 56))
                expert_w3_scale = state_dict.get(f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.experts_w3.{expert_index}.scale") 
                expert_w3_scales.append(expert_w3_scale.view(1, 16, 56))

            all_w1_weights = torch.cat(expert_w1_weights, dim=0) # all_w1_weights torch.Size([257, 2048, 7168]) torch.float8_e4m3fn
            all_w3_weights = torch.cat(expert_w3_weights, dim=0) # all_w3_weights torch.Size([257, 2048, 7168]) torch.float8_e4m3fn
            all_w1_scales = torch.cat(expert_w1_scales, dim=0)
            all_w3_scales = torch.cat(expert_w3_scales, dim=0)
            # print("all_w1_weights",all_w1_weights.shape, all_w1_weights.dtype)
            # print("all_w3_weights",all_w3_weights.shape, all_w3_weights.dtype)
            # print("all_w1_scales",all_w1_scales.shape, all_w1_scales.dtype)
            # print("all_w3_scales",all_w3_scales.shape, all_w3_scales.dtype)

            gpus_w1_weights = torch.chunk(all_w1_weights, 8, dim=1) # [8][257, 256, 7168]
            gpus_w3_weights = torch.chunk(all_w3_weights, 8, dim=1)
            gpus_w1_scales = torch.chunk(all_w1_scales, 8, dim=1)
            gpus_w3_scales = torch.chunk(all_w3_scales, 8, dim=1)
            # print("gpus_w1_scales",gpus_w1_scales[0].shape, gpus_w1_scales[0].dtype)
            # print("gpus_w3_scales",gpus_w3_scales[0].shape, gpus_w3_scales[0].dtype)
            for device_id in range(8):
                # [257, 128, 4, 7168]
                weight = torch.cat([gpus_w1_weights[device_id].view(257, 128, 2, 7168),gpus_w3_weights[device_id].view(257, 128, 2, 7168)], dim =2)
                # [257, 512, 7168]
                weight = weight.view(257,512,7168)
                # [257, 2, 2, 56] torch.float32
                scale = torch.cat([gpus_w1_scales[device_id].view(257,2,1,56), gpus_w3_scales[device_id].view(257,2,1,56)],dim=2)
                scale = scale.view(257,4,56)
                # print("scale", scale.shape,scale.dtype)
                experts_scales_swizzled = torch.zeros(257, 4, 64)
                for i in range(64):
                    experts_scales_swizzled[..., i] = scale[..., ((i % 8) * 8 + i // 8) % 56]
                if ((i % 8) * 8 + i // 8) >= 56:
                    experts_scales_swizzled[..., i] = 0.0
                experts_scales_swizzled = experts_scales_swizzled.to(torch.bfloat16)
                # print("experts_scales_swizzled", experts_scales_swizzled.shape,experts_scales_swizzled.dtype)
                gpus_[device_id][f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.weight"]= weight 
                gpus_[device_id][f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.experts_scales_swizzled"]= experts_scales_swizzled 
                gpus_[device_id][f"layers.{layer_id}.ffn.routed_up_gate_silu.up_gate_silu.bias"] = bias
            # print(gpus_w1_weights[0].shape)
            # print(gpus_w1_weights[1].shape)

            # print("bias ", bias.shape, bias.dtype)
            # print("shared_experts_w1_weight",shared_experts_w1_weight.shape, shared_experts_w1_weight.dtype)
            # print("shared_experts_w1_scale",shared_experts_w1_scale.shape, shared_experts_w1_scale.dtype)
            # print("shared_experts_w3_weight",shared_experts_w3_weight.shape, shared_experts_w3_weight.dtype)
            # print("shared_experts_w3_scale",shared_experts_w3_scale.shape, shared_experts_w3_scale.dtype)
    return gpus_
# 
def MoEDown(state_dict: dict):
    # dev_mat_in[dev_id] torch.Size([257, 7168, 256]) torch.float8_e4m3fn
    # dev_mat_scale[dev_id] torch.Size([257, 56, 2]) torch.bfloat16
    # mat_scale_tilert torch.Size([257, 1024, 2]) torch.bfloat16
    # for key, value in state_dict.items():
    #     if ".down.w2" in key:
    #         print(key)
    gpus_ = {i: {} for i in range(8)}
    for layer_id in range(0,67):
        shared_expert_w2_weight = state_dict.get(f"layers.{layer_id}.ffn.down.shared_experts_w2.weight") # shared_expert_w2_weight torch.Size([7168, 2048]) torch.float8_e4m3fn
        shared_expert_w2_scale = state_dict.get(f"layers.{layer_id}.ffn.down.shared_experts_w2.scale") # shared_expert_w2_scale torch.Size([56, 16]) torch.float32
        if (shared_expert_w2_weight is not None and shared_expert_w2_scale is not None):
            # print("shared_expert_w2_weight",shared_expert_w2_weight.shape, shared_expert_w2_weight.dtype)
            # print("shared_expert_w2_scale",shared_expert_w2_scale.shape, shared_expert_w2_scale.dtype)
            expert_w2_weights = [shared_expert_w2_weight.view(1, 7168, 2048)]
            expert_w2_scales = [shared_expert_w2_scale.view(1, 56, 16)]
            for expert_index in range(256):
                expert_w2_weight = state_dict.get(f"layers.{layer_id}.ffn.down.experts_w2.{expert_index}.weight") 
                expert_w2_scale = state_dict.get(f"layers.{layer_id}.ffn.down.experts_w2.{expert_index}.scale") 
                expert_w2_weights.append(expert_w2_weight.view(1, 7168, 2048)) 
                expert_w2_scales.append(expert_w2_scale.view(1, 56, 16))
            all_w2_weights = torch.cat(expert_w2_weights, dim=0) 
            all_w2_scales = torch.cat(expert_w2_scales, dim=0)
            gpus_w2_weights = torch.chunk(all_w2_weights, 8, dim=2) # 8*[257, 7168, 256]
            gpus_w2_scales = torch.chunk(all_w2_scales, 8, dim=2) # 8*[257, 56, 2]
            # print("gpus_w2_weights", gpus_w2_weights[0].shape)
            # print("gpus_w2_scales", gpus_w2_scales[0].shape)
            for device_id in range(8):
                dev_mat_in = gpus_w2_weights[device_id]
                mat_scale_tilert = gpus_w2_scales[device_id]
                # repeat to 896 and then pad to 1024
                mat_scale_tilert = (
                    mat_scale_tilert.reshape(257, 56, 1, 2).repeat(1, 1, 16, 1).reshape(257, 128, 14)
                )
                padding_zeros = torch.zeros((257, 128, 2), dtype=torch.bfloat16, device=mat_scale_tilert.device)
                mat_scale_tilert = torch.cat([mat_scale_tilert, padding_zeros], dim=2)
                mat_scale_tilert = mat_scale_tilert.reshape(257, 1024, 2).to(torch.bfloat16)
                # dev_mat_in torch.Size([257, 7168, 256]) torch.float8_e4m3fn
                # mat_scale_tilert torch.Size([257, 1024, 2]) torch.bfloat16
                # print("dev_mat_in", dev_mat_in.shape, dev_mat_in.dtype)
                # print("mat_scale_tilert", mat_scale_tilert.shape, mat_scale_tilert.dtype)
                gpus_[device_id][f"layers.{layer_id}.ffn.down.experts_w2_weight"]= dev_mat_in 
                gpus_[device_id][f"layers.{layer_id}.ffn.down.experts_w2_scale"]= mat_scale_tilert 
    return gpus_

def main():
    state_dicts = load_origin_state_dicts("/data2/shared/deepseekv3.1", 1, 256)
    convert_state_dict = convert_state_dict_origin2tilert(state_dicts[0])
    # print(convert_state_dict.keys())
    # gpuss_ = RMSNormProjectQKVWaRope_weight_exchange(convert_state_dict)
    # print(gpuss_[0])
    # gpus_rmsnorm_proj_qwb_rope =  RMSNormProjectQWbRoPE_weight_exchange(convert_state_dict)
    # gpus_kvrmsnorm = KVRMSNorm_weight_exchange(convert_state_dict)
    # print(gpus_kvrmsnorm[1].keys())
    # temp = ProjectQWb_weight_exchange(convert_state_dict)
    # print(temp[0].keys())
    # temp = Unproj_O_Allreduce_weight_exchange(convert_state_dict)
    # temp = Unproj_Owb(convert_state_dict)
    # print(temp[0].keys())
    # temp = MoeRoutedUpGateSilu_weight_exchange(convert_state_dict)
    temp =  MoEDown(convert_state_dict)
    # print(temp[3].keys())
    # print(temp[0].keys())
    # temp =  RMSNormMLPUpGateSiLU_weight_exchange(convert_state_dict)
    # temp = Down_Allreduce_weight_exchange(convert_state_dict)
    # temp = RMSNormExpertProj_weight_exchange(convert_state_dict) 
    # print(temp[0].keys())
    # print(temp[1]["layers.1.ffn.down.mat_scale_tilert"].dtype)
    # print(gpus_rmsnorm_proj_qwb_rope[0].keys())
    # test_e2e_forward_pass("configs/config_671B_layer2_device1.json")

    # 统计数量
    # from collections import defaultdict
    # layer_counts_detailed = defaultdict(int)
    
    # for key, value in convert_state_dict.items():
    #     # 提取前两级或三级作为层名
    #     parts = key.split('.')
    #     if len(parts) >= 2:
    #         layer_name = '.'.join(parts[:2])  # 例如 "model.layers"
    #     else:
    #         layer_name = parts[0]
    #     layer_counts_detailed[layer_name] += 1

    # print("\n详细的层级统计:")
    # for layer, count in sorted(layer_counts_detailed.items()):
    #     print(f"{layer}: {count} 个参数")


if __name__ == "__main__":
    main()