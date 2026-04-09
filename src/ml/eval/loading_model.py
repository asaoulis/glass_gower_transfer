import os
import glob
import re


def find_best_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint-epoch=*-val_log_prob=*.ckpt"))
    
    best_checkpoint = None
    best_val_loss = float("inf")
    
    # Updated regex pattern to match any number of digits for epoch and allow negative float for loss
    loss_pattern = re.compile(r"checkpoint-epoch=(\d+)-val_log_prob=(-?\d+\.\d+).ckpt")
    
    for ckpt in checkpoint_files:
        match = loss_pattern.search(ckpt)
        if match:
            epoch = int(match.group(1))  # Extract epoch number
            val_loss = float(match.group(2))  # Extract validation loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = ckpt
    
    return best_checkpoint, best_val_loss

import os

def get_best_checkpoint(experiment_path, match_string):
    print(f"Searching recursively for best checkpoints in: {experiment_path}")
    best_checkpoints = []
    val_losses = []
    
    # os.walk explores the experiment_path and all of its subdirectories
    for root, dirs, files in os.walk(experiment_path):
        # 'root' is the current directory being evaluated 
        # (e.g., '/this/is/a/path_0')
        run_folder_name = os.path.basename(os.path.normpath(root))
        
        # Check if the match_string is in this specific folder's name
        if match_string not in run_folder_name:
            continue
            
        # If it matches, pass the full path (root) to your helper function
        best_checkpoint, best_val_loss = find_best_checkpoint(root)
        
        # Protect against appending None if find_best_checkpoint comes up empty
        if best_checkpoint is not None:
            best_checkpoints.append(best_checkpoint)
            val_losses.append(best_val_loss)
            
    print("Best checkpoints found:", best_checkpoints)
    return best_checkpoints, val_losses
