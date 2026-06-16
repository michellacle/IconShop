#!/usr/bin/env python3
"""stdin→stdout SVG generation wrapper for IconShop."""

import sys
import os
import argparse

import torch
import numpy as np

from model.decoder import SketchDecoder
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BBOX = 200
TOKENIZER_NAME = "google/bert_uncased_L-12_H-512_A-8"
TEXT_LEN = 50


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_model(weight_path, pix_len, device):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    config = {
        "hidden_dim": 1024,
        "embed_dim": 512,
        "num_layers": 16,
        "num_heads": 8,
        "dropout_rate": 0.1,
    }

    decoder = SketchDecoder(
        config=config,
        pix_len=pix_len,
        text_len=TEXT_LEN,
        num_text_token=tokenizer.vocab_size,
        word_emb_path="ckpts/word_embedding_512.pt",
        pos_emb_path=None,
    )

    ckpt_file = os.path.join(weight_path, "model.safetensors")
    if os.path.exists(ckpt_file):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_file)
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    else:
        ckpt_file = os.path.join(weight_path, "pytorch_model.bin")
        state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    filtered = {k: v for k, v in state_dict.items()
                if k not in ("pos_embed.position", "pos_embed.pos_embed.weight")}
    decoder.load_state_dict(filtered, strict=False)

    if "pos_embed.position" in state_dict:
        src = state_dict["pos_embed.position"]
        dst = decoder.pos_embed.position.data
        n = min(src.shape[0], dst.shape[0])
        dst[:n] = src[:n]

    if "pos_embed.pos_embed.weight" in state_dict:
        src = state_dict["pos_embed.pos_embed.weight"]
        dst = decoder.pos_embed.pos_embed.weight.data
        n = min(src.shape[0], dst.shape[0])
        dst[:n] = src[:n]

    decoder = decoder.to(device).eval()
    return decoder, tokenizer


def raster_svg(pixels):
    pixels = np.array(pixels)
    if pixels.ndim != 2 or pixels.shape[1] < 2:
        return []

    pixels = pixels - 6

    svg_tensors = []
    path_tensor = []
    i = 0
    while i < len(pixels):
        cmd = int(pixels[i][0])
        if cmd == -3:  # Move
            if i + 2 >= len(pixels):
                break
            cmd_tensor = np.zeros(9)
            cmd_tensor[0] = 0
            cmd_tensor[7:9] = pixels[i + 2]
            start_pos = pixels[i + 1]
            end_pos = pixels[i + 2]
            if np.all(start_pos == end_pos) and path_tensor:
                svg_tensors.append(torch.tensor(path_tensor))
                path_tensor = []
            path_tensor.append(cmd_tensor.tolist())
            i += 3
        elif cmd == -2:  # Line
            if i + 1 >= len(pixels):
                break
            cmd_tensor = np.zeros(9)
            cmd_tensor[0] = 1
            cmd_tensor[7:9] = pixels[i + 1]
            path_tensor.append(cmd_tensor.tolist())
            i += 2
        elif cmd == -1:  # Curve
            if i + 3 >= len(pixels):
                break
            cmd_tensor = np.zeros(9)
            cmd_tensor[0] = 2
            cmd_tensor[3:5] = pixels[i + 1]
            cmd_tensor[5:7] = pixels[i + 2]
            cmd_tensor[7:9] = pixels[i + 3]
            path_tensor.append(cmd_tensor.tolist())
            i += 4
        else:
            i += 1

    if path_tensor:
        svg_tensors.append(torch.tensor(path_tensor))
    return [svg_tensors]


def pixels_to_svg_str(sample_pixels):
    """Convert model output pixels to SVG XML string. Returns None on failure."""
    for sample_pixel in sample_pixels:
        data_list = raster_svg(sample_pixel)
        for data in data_list:
            if not data or len(data[0]) == 0:
                continue
            try:
                paths = []
                for d in data:
                    path = SVGTensor.from_data(d)
                    path = SVG.from_tensor(path.data, viewbox=Bbox(BBOX))
                    path.fill_(True)
                    paths.append(path)
                groups = paths[0].svg_path_groups
                for p in paths[1:]:
                    groups.extend(p.svg_path_groups)
                svg = SVG(groups, viewbox=Bbox(BBOX))
                return svg.to_str()
            except Exception as e:
                log(f"render error: {e}")
    return None


def generate(decoder, tokenizer, prompt, n_samples, device, top_k=0, top_p=0.5):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=TEXT_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
    )
    tokens = encoded["input_ids"].squeeze()

    results = []
    bs = min(n_samples, 4)
    tokens_batch = tokens.repeat(bs, 1).to(device)
    remaining = n_samples

    while remaining > 0:
        batch = min(bs, remaining)
        batch_tokens = tokens_batch[:batch]
        sample_pixels = decoder.sample(n_samples=batch, text=batch_tokens,
                                       top_k=top_k, top_p=top_p)
        for px in sample_pixels:
            svg_str = pixels_to_svg_str([px])
            if svg_str:
                results.append(svg_str)
                remaining -= 1
                if remaining <= 0:
                    break

    return results


def main():
    parser = argparse.ArgumentParser(description="IconShop stdin→stdout CLI")
    parser.add_argument("--weight", type=str, required=True, help="checkpoint dir")
    parser.add_argument("--prompt", type=str, default=None, help="text prompt (reads stdin if omitted)")
    parser.add_argument("-n", type=int, default=1, help="number of SVGs to generate (default: 1)")
    parser.add_argument("--pix-len", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=0.5, help="nucleus sampling probability (0.0-1.0)")
    parser.add_argument("--top-k", type=int, default=0, help="top-k sampling (0=disabled)")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    log(f"loading model from {args.weight} on {device}...")
    decoder, tokenizer = load_model(args.weight, args.pix_len, device)
    log("model loaded")

    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        log("reading prompts from stdin (one per line, ctrl-d to end)...")
        prompts = [line.strip() for line in sys.stdin if line.strip()]

    for prompt in prompts:
        log(f"generating {args.n} SVG(s) for \"{prompt}\"...")
        svgs = generate(decoder, tokenizer, prompt, args.n, device,
                        top_k=args.top_k, top_p=args.top_p)
        for svg_str in svgs:
            sys.stdout.write(svg_str)
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
