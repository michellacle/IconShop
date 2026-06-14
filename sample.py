import os
import time
import torch
import argparse
import numpy as np

from model.decoder import SketchDecoder
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NUM_SAMPLE = 4
BS = 2
BBOX = 200


def sample(args, cfg):
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name'])

    model_config = {
        'hidden_dim': args.hidden_dim,
        'embed_dim': args.embed_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dropout_rate': args.dropout_rate,
    }

    sketch_decoder = SketchDecoder(
        config=model_config,
        pix_len=cfg['pix_len'],
        text_len=cfg['text_len'],
        num_text_token=tokenizer.vocab_size,
        word_emb_path=cfg['word_emb_path'],
        pos_emb_path=cfg['pos_emb_path'],
    )

    ckpt_file = os.path.join(args.sketch_weight, 'model.safetensors')
    if os.path.exists(ckpt_file):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_file)
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    else:
        ckpt_file = os.path.join(args.sketch_weight, 'pytorch_model.bin')
        state_dict = torch.load(ckpt_file, map_location='cpu', weights_only=False)

    # Filter pos_embed (handled separately for pix_len mismatch)
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

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    texts = args.prompts

    for text in texts:
        print(f'Generate SVG for "{text}"...')

        output_dir = os.path.join(args.output, text.replace(',', '_').replace(' ', '_'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        encoded_dict = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg['text_len'],
            add_special_tokens=True,
            return_token_type_ids=False,
        )
        tokenized_text = encoded_dict["input_ids"].squeeze()
        tokenized_text = tokenized_text.repeat(BS, 1).to(device)

        generated_svg = []
        start_time = time.time()
        while len(generated_svg) < NUM_SAMPLE:
            sample_pixels = sketch_decoder.sample(n_samples=BS, text=tokenized_text)
            generated_svg += sample_pixels
        end_time = time.time()
        print(f'Generate {len(generated_svg)} svg in {end_time - start_time:.1f} seconds')

        print('Rendering...')
        gen_data = []
        for si, sample_pixel in enumerate(generated_svg):
            print(f'  Sample {si}: shape={sample_pixel.shape}')
            gen_data += raster_svg(sample_pixel)

        print('Saving...')
        saved = 0
        for index, data in enumerate(gen_data):
            if not data or not data[0]:
                print(f'  Skipped sample {index}: empty path data')
                continue
            try:
                paths = []
                for d in data:
                    path = SVGTensor.from_data(d)
                    path = SVG.from_tensor(path.data, viewbox=Bbox(BBOX))
                    path.fill_(True)
                    paths.append(path)
                path_groups = paths[0].svg_path_groups
                for i in range(1, len(paths)):
                    path_groups.extend(paths[i].svg_path_groups)
                svg = SVG(path_groups, viewbox=Bbox(BBOX))
                outpath = os.path.join(output_dir, f'{str(index).zfill(5)}.svg')
                svg.save_svg(outpath)
                saved += 1
            except Exception as err_msg:
                print(f'  Failed sample {index}: {err_msg}')
                continue
        print(f'Saved {saved} SVGs to {output_dir}/')


def raster_svg(pixels):
    try:
        pixels = np.array(pixels)
        if pixels.ndim != 2 or pixels.shape[1] < 2:
            print(f'    Invalid pixel shape: {pixels.shape}')
            return []

        pixels = pixels - 6  # 3 END_TOKEN + 1 SVG_END + 2 CAUSAL_TOKEN

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
    except Exception as error_msg:
        print(f'    raster_svg error: {error_msg}')
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sketch_weight", type=str, required=True)
    parser.add_argument("--prompts", type=str, nargs='+', default=['star'])

    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    parser.add_argument("--pix_len", type=int, default=512)
    parser.add_argument("--num_sample", type=int, default=4)

    args = parser.parse_args()

    cfg = {
        'pix_len': args.pix_len,
        'text_len': 50,
        'tokenizer_name': 'google/bert_uncased_L-12_H-512_A-8',
        'word_emb_path': 'ckpts/word_embedding_512.pt',
        'pos_emb_path': None,
    }

    import __main__ as _m
    _m.NUM_SAMPLE = args.num_sample

    sample(args, cfg)
