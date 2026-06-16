#!/usr/bin/env python3
"""Generate SVG samples from each epoch checkpoint and build an HTML gallery."""
import os
import sys
import glob
import random
import time
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from transformers import AutoTokenizer

# Project imports
from model.decoder import SketchDecoder
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Config
BS = 2
BBOX = 200
NUM_SAMPLE = 8  # More samples for better percentile selection
PROMPTS = [
    # Original prompts
    "star",
    "heart",
    "tree",
    "house",
    "bicycle",
    "smile",
    # Nature & weather
    "sun",
    "moon",
    "cloud",
    "rain",
    "lightning",
    "snowflake",
    "flower",
    "leaf",
    "mountain",
    "cactus",
    "mushroom",
    # Animals
    "dog",
    "cat",
    "bird",
    "fish",
    "butterfly",
    "starfish",
    # Objects & tools
    "camera",
    "clock",
    "key",
    "lock",
    "bell",
    "scissors",
    "wrench",
    "compass",
    "map",
    "flag",
    "gift",
    "trophy",
    "target",
    "crown",
    "sword",
    "anchor",
    "rocket",
    "book",
    "envelope",
    "coffee",
    "umbrella",
    # Vehicles
    "car",
    "boat",
    "ship",
    "airplane",
    "train",
    # Food
    "apple",
    "banana",
    "pizza",
    # Music & instruments
    "music",
    "guitar",
    "piano",
    "drum",
    "microphone",
    "violin",
    "trumpet",
    # Body & anatomy
    "eye",
    "hand",
    "brain",
    "skull",
    "tooth",
    # Symbols & tech
    "wifi",
    "battery",
    "globe",
    "diamond",
    # Fantasy
    "ghost",
    "fire",
    "water",
    "storm",
    "wind",
]
CHECKPOINT_DIR = os.path.expanduser("~/code/IconShop/proj_log/FIGR_SVG")
OUTPUT_DIR = os.path.expanduser("~/code/IconShop/output/gallery")

CFG = {
    'pix_len': 512,
    'text_len': 50,
    'tokenizer_name': 'google/bert_uncased_L-12_H-512_A-8',
    'word_emb_path': 'ckpts/word_embedding_512.pt',
    'pos_emb_path': None,
}

MODEL_CONFIG = {
    'hidden_dim': 1024,
    'embed_dim': 512,
    'num_layers': 16,
    'num_heads': 8,
    'dropout_rate': 0.1,
}


def load_model(ckpt_dir, device):
    """Load a SketchDecoder from a checkpoint directory."""
    tokenizer = AutoTokenizer.from_pretrained(CFG['tokenizer_name'])
    sketch_decoder = SketchDecoder(
        config=MODEL_CONFIG,
        pix_len=CFG['pix_len'],
        text_len=CFG['text_len'],
        num_text_token=tokenizer.vocab_size,
        word_emb_path=CFG['word_emb_path'],
        pos_emb_path=CFG['pos_emb_path'],
    )

    ckpt_file = os.path.join(ckpt_dir, 'model.safetensors')
    if os.path.exists(ckpt_file):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_file)
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    else:
        ckpt_file = os.path.join(ckpt_dir, 'pytorch_model.bin')
        state_dict = torch.load(ckpt_file, map_location='cpu', weights_only=False)

    filtered_state = {k: v for k, v in state_dict.items()
                      if k not in ('pos_embed.position', 'pos_embed.pos_embed.weight')}
    sketch_decoder.load_state_dict(filtered_state, strict=False)

    if 'pos_embed.position' in state_dict:
        ckpt_pos = state_dict['pos_embed.position']
        model_pos = sketch_decoder.pos_embed.position.data
        trimmed = ckpt_pos[:model_pos.shape[0]] if ckpt_pos.shape[0] > model_pos.shape[0] else ckpt_pos
        sketch_decoder.pos_embed.position.data[:trimmed.shape[0]] = trimmed

    if 'pos_embed.pos_embed.weight' in state_dict:
        ckpt_emb = state_dict['pos_embed.pos_embed.weight']
        model_emb = sketch_decoder.pos_embed.pos_embed.weight.data
        trimmed = ckpt_emb[:model_emb.shape[0]] if ckpt_emb.shape[0] > model_emb.shape[0] else ckpt_emb
        sketch_decoder.pos_embed.pos_embed.weight.data[:trimmed.shape[0]] = trimmed

    sketch_decoder = sketch_decoder.to(device).eval()
    print(f"  Loaded model from {ckpt_dir}")
    return sketch_decoder, tokenizer


def raster_svg(pixels):
    """Convert pixel tokens to SVG path data."""
    try:
        pixels = np.array(pixels)
        if pixels.ndim != 2 or pixels.shape[1] < 2:
            return []
        pixels = pixels - 6
        svg_tensors = []
        path_tensor = []
        i = 0
        while i < len(pixels):
            pix = pixels[i]
            cmd = int(pix[0])
            if cmd == -3:  # Move
                if i + 2 >= len(pixels):
                    break
                cmd_tensor = np.zeros(9)
                cmd_tensor[0] = 0
                cmd_tensor[7:9] = pixels[i+2]
                start_pos = pixels[i+1]
                end_pos = pixels[i+2]
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
                cmd_tensor[7:9] = pixels[i+1]
                path_tensor.append(cmd_tensor.tolist())
                i += 2
            elif cmd == -1:  # Curve
                if i + 3 >= len(pixels):
                    break
                cmd_tensor = np.zeros(9)
                cmd_tensor[0] = 2
                cmd_tensor[3:5] = pixels[i+1]
                cmd_tensor[5:7] = pixels[i+2]
                cmd_tensor[7:9] = pixels[i+3]
                path_tensor.append(cmd_tensor.tolist())
                i += 4
            else:
                i += 1
        if path_tensor:
            svg_tensors.append(torch.tensor(path_tensor))
        return [svg_tensors]
    except Exception as e:
        print(f"    raster_svg error: {e}")
        return []


def generate_svg_string(svg_tensors, bbox_size=BBOX):
    """Convert SVG tensors to an SVG string."""
    if not svg_tensors or not svg_tensors[0]:
        return None
    try:
        paths = []
        for data in svg_tensors[0]:
            path = SVGTensor.from_data(data)
            svg_obj = SVG.from_tensor(path.data, viewbox=Bbox(bbox_size))
            svg_obj.fill_(True)
            paths.append(svg_obj)
        path_groups = paths[0].svg_path_groups
        for i in range(1, len(paths)):
            path_groups.extend(paths[i].svg_path_groups)
        svg_obj = SVG(path_groups, viewbox=Bbox(bbox_size))
        # Write to temp file, then read back
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
            tmp_path = f.name
        svg_obj.save_svg(tmp_path)
        with open(tmp_path, 'r') as f:
            svg_str = f.read()
        os.unlink(tmp_path)
        return svg_str
    except Exception as e:
        print(f"    SVG conversion error: {e}")
        return None


def simple_score(svg_str):
    """Simple heuristic score for SVG quality (path count, complexity)."""
    if not svg_str:
        return 0
    # Count paths as a proxy for complexity/quality
    path_count = svg_str.count('<path')
    # Check for reasonable viewBox
    has_viewbox = 'viewBox' in svg_str or 'viewBox' in svg_str
    # Score based on path count and structure
    score = path_count * 10
    if has_viewbox:
        score += 5
    return score


def main():
    device = torch.device("cuda:0")
    output_dir = os.path.expanduser(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Find available checkpoints
    epochs = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*")))
    checkpoints = [(os.path.basename(e), e) for e in epochs if os.path.exists(os.path.join(e, 'model.safetensors'))]
    
    if not checkpoints:
        print("No checkpoints found!")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints: {[c[0] for c in checkpoints]}")

    # Collect all results: {checkpoint_name: {prompt: [svg_strings]}}
    all_results = {}

    for ckpt_name, ckpt_path in checkpoints:
        if not os.path.exists(os.path.join(ckpt_path, 'model.safetensors')):
            print(f"Skipping {ckpt_name}: no checkpoint found")
            continue

        print(f"\n{'='*60}")
        print(f"Loading checkpoint: {ckpt_name}")
        print(f"{'='*60}")

        sketch_decoder, tokenizer = load_model(ckpt_path, device)
        ckpt_results = {}

        for text in PROMPTS:
            print(f"\n  Generating: \"{text}\"")
            encoded_dict = tokenizer(
                text, return_tensors="pt", padding="max_length",
                truncation=True, max_length=CFG['text_len'],
                add_special_tokens=True, return_token_type_ids=False,
            )
            tokenized_text = encoded_dict["input_ids"].squeeze().repeat(BS, 1).to(device)

            generated_svg = []
            start_time = time.time()
            while len(generated_svg) < NUM_SAMPLE:
                with torch.no_grad():
                    sample_pixels = sketch_decoder.sample(n_samples=BS, text=tokenized_text)
                generated_svg += sample_pixels
            elapsed = time.time() - start_time
            print(f"    Generated {len(generated_svg)} samples in {elapsed:.1f}s")

            svg_strings = []
            scores = []
            for si, sample_pixel in enumerate(generated_svg):
                gen_data = raster_svg(sample_pixel)
                svg_str = generate_svg_string(gen_data)
                if svg_str:
                    svg_strings.append(svg_str)
                    scores.append(simple_score(svg_str))

            print(f"    {len(svg_strings)}/{NUM_SAMPLE} valid SVGs")
            ckpt_results[text] = list(zip(svg_strings, scores))

        all_results[ckpt_name] = ckpt_results

    # Build HTML gallery
    print(f"\n{'='*60}")
    print("Building HTML gallery...")
    print(f"{'='*60}")

    html = build_html(all_results)
    html_path = os.path.join(output_dir, "gallery.html")
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"\nDone! Gallery saved to: {html_path}")


def build_html(all_results):
    """Build an HTML page with embedded SVGs organized by prompt, showing epoch progression."""
    prompts = [p for p in PROMPTS if any(p in all_results.get(ck, {}) for ck in all_results)]

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IconShop Training Gallery</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 20px;
  }
  h1 {
    text-align: center;
    color: #e94560;
    margin-bottom: 6px;
    font-size: 28px;
  }
  .subtitle {
    text-align: center;
    color: #888;
    margin-bottom: 16px;
    font-size: 13px;
  }
  #search {
    display: block;
    margin: 0 auto 24px;
    padding: 10px 16px;
    width: 100%;
    max-width: 400px;
    border-radius: 8px;
    border: 1px solid #0f3460;
    background: #16213e;
    color: #e0e0e0;
    font-size: 15px;
    outline: none;
  }
  #search:focus { border-color: #e94560; }
  .prompt-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 12px;
    padding: 12px;
    background: #16213e;
    border-radius: 10px;
    border: 1px solid #0f3460;
  }
  .prompt-label {
    min-width: 100px;
    font-weight: 700;
    color: #e94560;
    font-size: 14px;
    padding-top: 60px;
    text-align: right;
  }
  .epoch-cards {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    flex: 1;
  }
  .epoch-card {
    background: #0f3460;
    border-radius: 8px;
    padding: 8px;
    text-align: center;
    min-width: 140px;
    flex: 1;
  }
  .epoch-card .epoch-name {
    font-size: 11px;
    color: #e94560;
    font-weight: 600;
    margin-bottom: 4px;
  }
  .epoch-card svg {
    width: 120px;
    height: 120px;
  }
  .epoch-card .score {
    font-size: 10px;
    color: #888;
    margin-top: 2px;
  }
  .epoch-card.empty {
    opacity: 0.3;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 150px;
  }
  .epoch-card.empty .epoch-name { padding-top: 60px; }
  .prompt-row[data-prompt] { display: flex; }
  .prompt-row[data-prompt].hidden { display: none; }
</style>
</head>
<body>
<h1>IconShop Training Gallery</h1>
<p class="subtitle">Top SVG per prompt across epochs — use search to filter</p>
<input type="text" id="search" placeholder="Search prompts..." oninput="filterPrompts(this.value)">
<div id="gallery">
""")

    # Group by prompt, show best SVG per epoch side by side
    for prompt in prompts:
        html_parts.append(f'<div class="prompt-row" data-prompt="{prompt}">')
        html_parts.append(f'<div class="prompt-label">{prompt}</div>')
        html_parts.append(f'<div class="epoch-cards">')

        for ckpt_name in all_results:
            ckpt_data = all_results[ckpt_name]
            svg_score_pairs = ckpt_data.get(prompt, [])
            if not svg_score_pairs:
                html_parts.append(f'<div class="epoch-card empty"><div class="epoch-name">{ckpt_name}</div></div>')
                continue

            # Sort by score, pick top
            sorted_svgs = sorted(svg_score_pairs, key=lambda x: x[1], reverse=True)
            top_svg = sorted_svgs[0][0]
            top_score = sorted_svgs[0][1]

            if top_svg:
                html_parts.append(f'<div class="epoch-card">')
                html_parts.append(f'<div class="epoch-name">{ckpt_name}</div>')
                html_parts.append(top_svg)
                html_parts.append(f'<div class="score">score: {top_score}</div>')
                html_parts.append(f'</div>')
            else:
                html_parts.append(f'<div class="epoch-card empty"><div class="epoch-name">{ckpt_name}</div></div>')

        html_parts.append(f'</div></div>')

    html_parts.append("""</div>
<script>
function filterPrompts(query) {
  query = query.toLowerCase();
  document.querySelectorAll('.prompt-row').forEach(row => {
    const prompt = row.getAttribute('data-prompt').toLowerCase();
    row.classList.toggle('hidden', !prompt.includes(query));
  });
}
</script>
</body></html>""")
    return '\n'.join(html_parts)


if __name__ == "__main__":
    main()
