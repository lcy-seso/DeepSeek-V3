#!/usr/bin/env python3

import os
import json
import argparse
from typing import Optional
import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)
import torch.nn as nn


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


class SingleLayerModel(nn.Module):
    """Wrapper class for single layer inference using DeepSeek-V3 architecture"""

    def __init__(
        self, model_path: str, config, layer_index: int, device_map: str = "auto"
    ):
        super().__init__()

        # Import the model classes
        try:
            from model import Transformer as DeepSeekV3Transformer, ModelArgs
        except ImportError:
            raise ImportError(
                "Cannot import DeepSeek-V3 model classes. Make sure model.py is available."
            )

        # Create a minimal config for single layer
        model_args = ModelArgs(
            vocab_size=config.vocab_size,
            dim=config.hidden_size,
            inter_dim=getattr(config, "intermediate_size", config.hidden_size * 4),
            moe_inter_dim=getattr(config, "moe_intermediate_size", 256),
            n_layers=1,  # Only one layer
            n_dense_layers=1,
            n_heads=config.num_attention_heads,
            n_routed_experts=getattr(config, "num_experts", 256),
            n_shared_experts=getattr(config, "num_shared_experts", 1),
            n_activated_experts=getattr(config, "num_experts_per_tok", 8),
            n_expert_groups=getattr(config, "num_expert_groups", 8),
            n_limited_groups=getattr(config, "num_limited_groups", 4),
            route_scale=getattr(config, "route_scale", 2.5),
            score_func=getattr(config, "score_func", "sigmoid"),
            q_lora_rank=getattr(config, "q_lora_rank", 1536),
            kv_lora_rank=getattr(config, "kv_lora_rank", 512),
            qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 128),
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", 64),
            v_head_dim=getattr(config, "v_head_dim", 16),
            dtype="bf16",
        )

        print(f"Creating single layer model for layer {layer_index}...")

        # Create the single layer model
        self.model = DeepSeekV3Transformer(model_args)

        # Load the specific layer weights from the full model
        print(f"Loading weights for layer {layer_index}...")
        self._load_layer_weights(model_path, layer_index, device_map)

        self.config = config
        self.layer_index = layer_index

        print(f"Successfully loaded layer {layer_index}")

    def _load_layer_weights(self, model_path: str, layer_index: int, device_map: str):
        """Load weights for the specific layer from the full model"""
        # Load the full model state dict
        full_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Extract the weights for the specific layer
        full_state_dict = full_model.state_dict()
        layer_state_dict = {}

        # Copy embedding and output layer weights
        for key, value in full_state_dict.items():
            if key.startswith("model.embed_tokens") or key.startswith("lm_head"):
                layer_state_dict[key] = value
            elif key.startswith(f"model.layers.{layer_index}"):
                # Rename to layer 0 for our single layer model
                new_key = key.replace(f"model.layers.{layer_index}", "model.layers.0")
                layer_state_dict[new_key] = value
            elif key.startswith("model.norm"):
                layer_state_dict[key] = value

        # Load the weights into our single layer model
        self.model.load_state_dict(layer_state_dict, strict=False)

        # Clean up the full model to save memory
        del full_model
        torch.cuda.empty_cache()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Use the model's forward method
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)


def load_single_layer(
    model_path: str, config, layer_index: int, device_map: str = "auto"
):
    """Load only a specific layer from the model"""
    return SingleLayerModel(model_path, config, layer_index, device_map)


def load_model_and_tokenizer(
    model_path: str,
    device_map: str = "auto",
    layer_index: Optional[int] = None,
):
    print(f"Loading model from {model_path}")

    if layer_index is not None:
        print(f"Loading only layer {layer_index}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    generation_config = GenerationConfig(
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if layer_index is not None:
        # Load only specific layer
        model = load_single_layer(model_path, config, layer_index, device_map)
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    return model, tokenizer, generation_config


def generate_response(
    model,
    tokenizer,
    prompt: str,
    generation_config,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
):
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    generation_config.max_new_tokens = max_new_tokens
    generation_config.temperature = temperature

    with torch.no_grad():
        # Check if it's a single layer model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Single layer model - use custom generation
            outputs = generate_single_layer(
                model, inputs, generation_config, tokenizer.eos_token_id
            )
        else:
            # Full model - use standard generation
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                pad_token_id=tokenizer.eos_token_id,
            )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response


def generate_single_layer(model, inputs, generation_config, eos_token_id):
    """Custom generation for single layer model"""
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # Simple greedy generation for single layer
    generated_ids = input_ids.clone()

    for _ in range(generation_config.max_new_tokens):
        # Get logits from the model
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Get last token logits

            # Apply temperature
            if generation_config.temperature > 0:
                logits = logits / generation_config.temperature

            # Sample next token
            if generation_config.do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        ),
                    ],
                    dim=-1,
                )

            # Stop if EOS token is generated
            if next_token.item() == eos_token_id:
                break

    return generated_ids


def interactive_chat(
    model, tokenizer, generation_config, max_new_tokens=200, temperature=0.7
):
    print("DeepSeek-V3 Interactive Chat (Press 'quit' to exit)")
    print("=" * 50)

    messages = []

    while True:
        try:
            user_input = input("\nUser: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            prompt = ""
            for msg in messages[-6:]:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                else:
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant: "

            print("Assistant: ", end="", flush=True)

            response = generate_response(
                model,
                tokenizer,
                prompt,
                generation_config,
                max_new_tokens,
                temperature,
            )

            print(response)
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_inference(
    model, tokenizer, generation_config, input_file: str, output_file: str
):
    with open(input_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    results = []
    for i, prompt in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}: {prompt[:50]}...")

        response = generate_response(model, tokenizer, prompt, generation_config)

        results.append({"prompt": prompt, "response": response})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V3 Hugging Face Inference")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument("--input-file", type=str, help="Input file for batch inference")
    parser.add_argument(
        "--output-file", type=str, help="Output file for batch inference"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--layer-index", type=int, help="Load only a specific layer (0-based index)"
    )

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"Running on rank {rank}/{world_size}")
        print(f"Model path: {args.model_path}")

    model, tokenizer, generation_config = load_model_and_tokenizer(
        args.model_path, args.device_map, args.layer_index
    )

    if rank == 0:
        print("Model loaded successfully!")

        if args.interactive:
            interactive_chat(
                model,
                tokenizer,
                generation_config,
                args.max_new_tokens,
                args.temperature,
            )
        elif args.input_file:
            output_file = args.output_file or "output.json"
            batch_inference(
                model, tokenizer, generation_config, args.input_file, output_file
            )
        else:
            prompt = "hello, how are you?"
            print(f"Prompt: {prompt}")
            response = generate_response(
                model,
                tokenizer,
                prompt,
                generation_config,
                args.max_new_tokens,
                args.temperature,
            )
            print(f"Response: {response}")


if __name__ == "__main__":
    main()
