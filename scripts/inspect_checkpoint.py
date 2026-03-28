#!/usr/bin/env python3

import argparse
from collections import OrderedDict

import torch
import torch.nn as nn


def _preview_tensor_keys(state_dict, limit=10):
    keys = list(state_dict.keys())
    sample = keys[:limit]
    print(f"Sample parameter keys ({len(sample)} of {len(keys)}):")
    for key in sample:
        value = state_dict[key]
        shape = tuple(value.shape) if torch.is_tensor(value) else type(value)
        print(f"  - {key}: {shape}")


def inspect_checkpoint(path):
    print(f"Inspecting: {path}")

    try:
        scripted = torch.jit.load(path, map_location="cpu")
        print("Type: TorchScript checkpoint")
        if isinstance(scripted, torch.jit.ScriptModule):
            print("Run with: model = torch.jit.load(path)")
        return
    except Exception:
        pass

    obj = torch.load(path, map_location="cpu")
    print(f"Python object type: {type(obj)}")

    if isinstance(obj, nn.Module):
        print("Type: Full nn.Module checkpoint (torch.save(model, ...))")
        print("Run with: model = torch.load(path)")
        return

    if isinstance(obj, (dict, OrderedDict)):
        keys = list(obj.keys())
        print(f"Top-level keys ({len(keys)} total): {keys[:20]}")

        if "model" in obj and isinstance(obj["model"], (dict, OrderedDict)):
            print("Type: Detectron2-style checkpoint (weights under key 'model')")
            _preview_tensor_keys(obj["model"])
            print("Suggested use: instantiate architecture, then load obj['model'] as state_dict")
            return

        if len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values()):
            print("Type: Raw state_dict checkpoint (param_name -> tensor)")
            _preview_tensor_keys(obj)
            print("Suggested use: instantiate architecture, then model.load_state_dict(obj)")
            return

        print("Type: Dict checkpoint (custom format)")
        print("Inspect top-level keys above and choose which sub-dict contains model weights.")
        return

    print("Type: Unknown/custom checkpoint format")


def main():
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint and classify its format.")
    parser.add_argument("checkpoint", nargs="?", default=None, type=str, help="Path to checkpoint file (.pth/.pt)")
    parser.add_argument("-c", "--checkpoint", dest="checkpoint_flag", default=None, type=str,
                        help="Path to checkpoint file (.pth/.pt)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_flag if args.checkpoint_flag else args.checkpoint
    if not checkpoint_path:
        parser.error("please provide a checkpoint path as positional arg or with --checkpoint")

    inspect_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()
