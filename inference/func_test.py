import pytest
import torch
import torch.distributed as dist
import re
import os

import json
from model import ModelArgs

import pdb

from model import Transformer as DeepSeekV3Transformer
from tilert.models.deepseek_v3.model import Transformer as TilertDeepSeekV3Transformer


@pytest.fixture(autouse=True)
def setup():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.set_default_device(f"cuda:{local_rank}")
    else:
        torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)


def convert_state_dict(tilert_state_dict: dict) -> dict:
    """
    Convert the state dict of the tilert's DeepSeekV3 model to the state dict of
    the original DeepSeekV3 model.
    """
    key_casting_maps = {
        # RMSNorm weight
        r"layers\.(\d+)\.ffn\.norm_up_gate\.norm_weight": r"layers.\1.ffn_norm.weight",
        r"layers\.(\d+)\.ffn\.rmsnorm_expert\.norm_weight": r"layers.\1.ffn_norm.weight",
        # MLP weight
        r"layers\.(\d+)\.ffn\.norm_up_gate.w1.weight": r"layers.\1.ffn.w1.weight",
        r"layers\.(\d+)\.ffn\.down.w2.weight": r"layers.\1.ffn.w2.weight",
        r"layers\.(\d+)\.ffn\.norm_up_gate.w3.weight": r"layers.\1.ffn.w3.weight",
        r"layers\.(\d+)\.ffn\.norm_up_gate.w1.scale": r"layers.\1.ffn.w1.scale",
        r"layers\.(\d+)\.ffn\.down.w2.scale": r"layers.\1.ffn.w2.scale",
        r"layers\.(\d+)\.ffn\.norm_up_gate.w3.scale": r"layers.\1.ffn.w3.scale",
        # Project weight
        r"layers\.(\d+)\.ffn\.rmsnorm_expert\.proj_weight": r"layers.\1.ffn.gate.weight",
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.route_gate\.bias": r"layers.\1.ffn.gate.bias",
        # Expert weight
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w1\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w1.weight",
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w1\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w1.scale",
        r"layers\.(\d+)\.ffn\.down.experts_w2\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w2.weight",
        r"layers\.(\d+)\.ffn\.down.experts_w2\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w2.scale",
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w3\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w3.weight",
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w3\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w3.scale",
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w1.weight": r"layers.\1.ffn.shared_experts.w1.weight",
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w1.scale": r"layers.\1.ffn.shared_experts.w1.scale",
        r"layers\.(\d+)\.ffn\.down.shared_experts_w2.weight": r"layers.\1.ffn.shared_experts.w2.weight",
        r"layers\.(\d+)\.ffn\.down.shared_experts_w2.scale": r"layers.\1.ffn.shared_experts.w2.scale",
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w3.weight": r"layers.\1.ffn.shared_experts.w3.weight",
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w3.scale": r"layers.\1.ffn.shared_experts.w3.scale",
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


@pytest.mark.parametrize("model_config", [])
def test_e2e_forward_pass(model_config):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    with open(model_config, "r", encoding="utf-8") as f_json:
        model_config = json.load(f_json)

    model_args = ModelArgs(**model_config)

    # Initialize models with adjusted config
    tilert_model = TilertDeepSeekV3Transformer(model_args, enable_tilert=False)
    origin_model = DeepSeekV3Transformer(model_args)

    if world_size > 1:
        dist.barrier()

    # origin_model.load_state_dict(convert_state_dict(tilert_model.state_dict()))

    # keys = [k for k in origin_model.state_dict().keys() if "ffn_norm.weight" in k]
    # print("\n".join(keys))

    # print(tilert_model.state_dict().keys())
    # print(origin_model.state_dict())

    x = torch.randint(0, model_args.vocab_size, (1, 1))
    ref_output = origin_model(x, start_pos=127)
    # tilert_output = tilert_model(x, start_pos=127)

    if rank == 3:
        print(f"rank-{rank}, x:", x)
        print(f"rank-{rank}, Ref:", ref_output)

    # abs_err = torch.abs(ref_output - tilert_output)
    # rel_err = abs_err / torch.abs(ref_output)
    # print(f"Rel err: max-{rel_err.max():.6f}/mean-{rel_err.mean():.6f}")
    # print(f"Abs err: max-{abs_err.max():.6f}/mean-{abs_err.mean():.6f}")

    # print("Tilert:", tilert_output)
    # cos_sim = torch.nn.functional.cosine_similarity(
    #     ref_output.flatten(), tilert_output.flatten(), dim=0
    # )
    # print(f"Cosine similarity: {cos_sim.item():.6f}")


def init_distributed():
    """Initialize distributed training"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=world_rank,
            init_method="env://",
            device_id=local_rank,
        )
    return local_rank, world_rank, world_size


def main():
    local_rank, world_rank, world_size = init_distributed()

    torch.set_default_device(f"cuda:{local_rank}")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)

    test_e2e_forward_pass("configs/config_671B_layer2_device1.json")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
