#!/usr/bin/env python3

import torch
import argparse
from generate_hf import load_model_and_tokenizer, generate_response


def test_single_layer():
    """Test single layer loading and generation"""

    # Model path
    model_path = "/data2/shared/dpsk_weights/DeepSeek-V3.1-Terminus"
    layer_index = 0  # Test with first layer

    print(f"Testing single layer loading for layer {layer_index}")
    print(f"Model path: {model_path}")

    try:
        # Load single layer model
        model, tokenizer, generation_config = load_model_and_tokenizer(
            model_path, "auto", layer_index
        )

        print("Single layer model loaded successfully!")

        # Test generation
        prompt = "Hello, how are you?"
        print(f"Testing generation with prompt: {prompt}")

        response = generate_response(
            model, tokenizer, prompt, generation_config, max_new_tokens=50
        )

        print(f"Generated response: {response}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_single_layer()
