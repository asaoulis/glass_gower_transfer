"""Checkpoint lookup: find_best_checkpoint and get_best_checkpoint pick the best file in a run folder by validation log-prob."""
import os
import glob
import re


def find_best_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint-epoch=*-val_log_prob=*.ckpt"))
    
    best_checkpoint = None
    best_val_loss = float("inf")
    
    loss_pattern = re.compile(r"checkpoint-epoch=(\d+)-val_log_prob=(-?\d+\.\d+).ckpt")
    
    for ckpt in checkpoint_files:
        match = loss_pattern.search(ckpt)
        if match:
            epoch = int(match.group(1))
            val_loss = float(match.group(2))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = ckpt
    
    return best_checkpoint, best_val_loss

import os

def get_best_checkpoint(experiment_path, match_string):
    print(f"Searching recursively for best checkpoints in: {experiment_path}")
    best_checkpoints = []
    val_losses = []
    
    for root, dirs, files in os.walk(experiment_path):
        run_folder_name = os.path.basename(os.path.normpath(root))

        if match_string not in run_folder_name:
            continue

        best_checkpoint, best_val_loss = find_best_checkpoint(root)

        if best_checkpoint is not None:
            best_checkpoints.append(best_checkpoint)
            val_losses.append(best_val_loss)

    # Sorted by val_loss so [0] is the global best. os.walk order is not.
    if val_losses:
        order = sorted(range(len(best_checkpoints)), key=lambda i: val_losses[i])
        best_checkpoints = [best_checkpoints[i] for i in order]
        val_losses = [val_losses[i] for i in order]
        if len(best_checkpoints) > 1:
            print(
                f"[get_best_checkpoint] {len(best_checkpoints)} run folders matched "
                f"'{match_string}'; using the global best (val={val_losses[0]}). "
                f"Others: {list(zip(best_checkpoints[1:], val_losses[1:]))}"
            )

    print("Best checkpoints found:", best_checkpoints)
    return best_checkpoints, val_losses
