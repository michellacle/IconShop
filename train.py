import os
import torch
import argparse
from dataset import SketchData
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np 
import sys
sys.path.insert(0, 'utils')

from transformers import get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator
from transformers import AutoTokenizer

from model.decoder import SketchDecoder


def train(args, cfg):
    accum_step = cfg['gradient_accumulation_steps']
    accelerator = Accelerator(gradient_accumulation_steps=accum_step)
    
    # Initialize dataset loader
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name'])
    train_dataset = SketchData(args.train_meta_file, args.svg_folder, args.maxlen, cfg['text_len'], tokenizer, require_aug=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                             shuffle=True,
                                             batch_size=args.batchsize,
                                             num_workers=args.num_workers,
                                             pin_memory=True)

    val_dataset = SketchData(args.val_meta_file, args.svg_folder, args.maxlen, cfg['text_len'], tokenizer, require_aug=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 shuffle=False,
                                                 batch_size=args.batchsize,
                                                 num_workers=args.num_workers)

    set_seed(2023)

    model = SketchDecoder(
        config={
            'hidden_dim': cfg['hidden_dim'],
            'embed_dim': cfg['embed_dim'], 
            'num_layers': cfg['num_layers'], 
            'num_heads': cfg['num_heads'],
            'dropout_rate': cfg['dropout_rate'],
        },
        pix_len=train_dataset.maxlen_pix,
        text_len=cfg['text_len'],
        num_text_token=tokenizer.vocab_size,
        word_emb_path=cfg['word_emb_path'],
        pos_emb_path=cfg['pos_emb_path'],
    )
   
    lr = cfg['lr'] * accelerator.num_processes
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = cfg['warm_up_steps'],
        num_training_steps = len(train_dataloader) * cfg['epoch']
    )

    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )

    num_update_steps_per_epoch = len(train_dataloader) // accum_step

    # logging 
    if accelerator.is_local_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, args.project_name))

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", ""))
        else:
            raise ValueError("Only support resuming from epoch checkpoints")

    # When resuming, --epochs means "train N more epochs"
    if starting_epoch > 0:
        cfg['epoch'] = starting_epoch + cfg['epoch']
        accelerator.print(f"Resumed from epoch {starting_epoch}, will train {cfg['epoch'] - starting_epoch} more epoch(s)")

    accelerator.print('Start training...')
    
    for epoch in range(starting_epoch, cfg['epoch']):
        model = model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch + 1}")
        total_loss, total_pix_loss, total_text_loss = 0., 0., 0.

        for pix, xy, mask, text in train_dataloader:
            with accelerator.accumulate(model):
                loss, pix_loss, text_loss = model(pix, xy, mask, text, return_loss=True)
                total_loss += loss.item() / accum_step
                total_pix_loss += pix_loss.item() / accum_step
                total_text_loss += text_loss.item() / accum_step
            
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradient
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients and accelerator.is_local_main_process:
                if overall_step % cfg['log_every'] == 0:
                    writer.add_scalar("loss/total_loss", total_loss, overall_step)
                    writer.add_scalar("loss/pix_loss", total_pix_loss, overall_step)
                    writer.add_scalar("loss/text_loss", total_text_loss, overall_step)
                    writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], overall_step)

                # Step-based checkpointing (saves every N steps)
                if cfg.get('save_every_steps', 0) > 0 and overall_step % cfg['save_every_steps'] == 0:
                    ckpt_path = os.path.join(args.output_dir, args.project_name, f'step_{overall_step}')
                    try:
                        accelerator.save_state(ckpt_path)
                        accelerator.print(f'Saved checkpoint at step {overall_step}')
                    except RuntimeError as e:
                        if 'CUDA' in str(e) or 'driver' in str(e).lower():
                            accelerator.print(f'WARNING: Checkpoint save skipped: {e}')
                        else:
                            raise

                total_loss, total_pix_loss, total_text_loss = 0., 0., 0.
                progress_bar.update(1)
                overall_step += 1

        progress_bar.close()
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            writer.flush()

        # save model after n epoch
        if (epoch+1) % cfg['save_every'] == 0:
            if accelerator.is_local_main_process:
                ckpt_path = os.path.join(args.output_dir, args.project_name, f'epoch_{epoch+1}')
                try:
                    accelerator.save_state(ckpt_path)
                except RuntimeError as e:
                    if 'CUDA' in str(e) or 'driver' in str(e).lower():
                        accelerator.print(f'WARNING: Checkpoint save skipped due to CUDA driver issue: {e}')
                    else:
                        raise

        # Validation loss 
        if (epoch+1) % cfg['val_every'] == 0:
            model.eval()
            accelerator.print('Testing...')
            all_losses = []
            with tqdm(val_dataloader, unit="batch", disable=not accelerator.is_local_main_process) as batch_data:
                for pix, xy, mask, text in batch_data:
                    with torch.no_grad():
                        loss, pix_loss, text_loss = model(pix, xy, mask, text, return_loss=True)
                        all_targets = accelerator.gather_for_metrics(loss)
                        all_losses.append(all_targets.mean().item())
            valid_loss = np.array(all_losses).mean()
            accelerator.print(f'Epoch {epoch + 1}: validation loss is {valid_loss}')

    if accelerator.is_local_main_process:
        writer.close()


if __name__ == "__main__":
    set_seed(2023)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_meta_file", type=str, required=True)
    parser.add_argument("--val_meta_file", type=str, required=True)
    parser.add_argument("--svg_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--batchsize", type=int, required=True)
    parser.add_argument("--maxlen", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=2, help="Data loader workers per GPU (keep low to avoid I/O lock-up)")
    parser.add_argument("--debug", action="store_true", default=False)

    # Model size overrides (for small test runs)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=16000)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--save_every_steps", type=int, default=0, help="Save checkpoint every N steps (0=disabled)")
    parser.add_argument("--val_every", type=int, default=5)

    args = parser.parse_args()

    config = {
        'tokenizer_name': 'google/bert_uncased_L-12_H-512_A-8',
        'text_len': 50,

        'hidden_dim': args.hidden_dim,
        'embed_dim': args.embed_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dropout_rate': args.dropout,
        'word_emb_path': 'ckpts/word_embedding_512.pt',
        'pos_emb_path': None,
        'gradient_accumulation_steps': args.gradient_accumulation,

        'lr': 3e-4,  # need scaling for different batch size
        'warm_up_steps': args.warmup_steps,
        'epoch': args.epochs,

        'log_every': args.log_every,
        'save_every': args.save_every,
        'save_every_steps': args.save_every_steps,
        'val_every': args.val_every,

        'batch_size': args.batchsize,
        'max_len': args.maxlen,
        'num_workers': args.num_workers,
    }

    # Create training folder
    result_folder = os.path.join(args.output_dir, args.project_name)
    os.makedirs(result_folder, exist_ok=True)

    with open(os.path.join(result_folder, 'config.json'), 'w') as f:
        import json
        json.dump(config, f, indent=4)
        
    train(args, config)
