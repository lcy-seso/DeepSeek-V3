import pytest
import torch
import re

import json
from model import ModelArgs

from model import Transformer as DeepSeekV3Transformer
from tilert.models.deepseek_v3.model import Transformer as TilertDeepSeekV3Transformer


@pytest.fixture(autouse=True)
def setup():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)


def convert_state_dict(tilert_state_dict: dict) -> dict:
    """
    Convert the state dict of the Tilert DeepSeekV3 model to the state dict of the DeepSeekV3 model.
    """
    key_casting_maps = {
        # RMSNorm weight
        r"layers\.(\d+)\.ffn\.norm\.weight": r"layers.\1.ffn_norm.weight",
        r"layers\.(\d+)\.ffn\.norm_up_gate.norm_weight": r"layers.\1.ffn_norm.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.route_proj\.norm_weight": r"layers.\1.ffn_norm.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_gate\.norm_weight": r"layers.\1.ffn_norm.weight",  # noqa: E501
        # MLP weight
        r"layers\.(\d+)\.ffn\.norm_up_gate.w1.weight": r"layers.\1.ffn.w1.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.norm_up_gate.w3.weight": r"layers.\1.ffn.w3.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.norm_up_gate.w1.scale": r"layers.\1.ffn.w1.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.norm_up_gate.w3.scale": r"layers.\1.ffn.w3.scale",  # noqa: E501
        # Project weight
        r"layers\.(\d+)\.ffn\.routed_gate\.proj_weight": r"layers.\1.ffn.gate.weight",  # noqa: E501
        # Expert weight
        r"layers\.(\d+)\.ffn\.experts_w1\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w1.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.experts_w1\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w1.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.experts_w2\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w2.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.experts_w2\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w2.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.experts_w3\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w3.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.experts_w3\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w3.scale",  # noqa: E501
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
    with open(model_config, "r", encoding="utf-8") as f_json:
        model_config = json.load(f_json)
    model_args = ModelArgs(**model_config)

    x = torch.randint(0, model_args.vocab_size, (1, 1))

    tilert_model = TilertDeepSeekV3Transformer(model_args)
    origin_model = DeepSeekV3Transformer(model_args)
    origin_model.load_state_dict(convert_state_dict(tilert_model.state_dict()))

    ref_output = origin_model(x, start_pos=127)
    tilert_output = tilert_model(x, start_pos=127)

    logit_sum = torch.sum(ref_output)
    logit_sum_tilert = torch.sum(tilert_output)
    print(logit_sum, logit_sum_tilert)

    assert torch.allclose(ref_output, tilert_output)


def main():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    test_e2e_forward_pass("configs/config_16B_v3.1.json")


if __name__ == "__main__":
    main()
