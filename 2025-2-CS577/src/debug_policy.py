import pickle
import torch
import numpy as np
import os
import sys

# Point to your folders
sys.path.append(os.path.join(os.getcwd(), 'src/demospeedup/aloha'))
sys.path.append(os.path.join(os.getcwd(), 'src/demospeedup/aloha/act'))

# --- PATCH START ---
# We must fake arguments before importing ACTPolicy
_real_argv = sys.argv.copy()
sys.argv = [
    'debug_policy.py', '--ckpt_dir', 'dummy', '--policy_class', 'ACT', 
    '--task_name', 'dummy', '--seed', '0', '--num_epochs', '1'
]

from act.policy import ACTPolicy

# Restore arguments
# sys.argv = _real_argv
# --- PATCH END ---

def debug(ckpt_dir):
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    
    # 1. Inspect Stats
    print(f"--- Inspecting {stats_path} ---")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    
    # 2. Setup Policy
    print("\n--- Running Dummy Inference ---")
    policy_config = {
        "lr": 1e-5, "num_queries": 50, "kl_weight": 10, 
        "hidden_dim": 512, "dim_feedforward": 3200,
        "lr_backbone": 1e-5, "backbone": "resnet18",
        "enc_layers": 4, "dec_layers": 7, "nheads": 8,
        "camera_names": ['top']
    }
    
    policy = ACTPolicy(policy_config)
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
        
    print(f"Loading weights from {ckpt_path}...")
    policy.load_state_dict(torch.load(ckpt_path))
    policy.cuda()
    policy.eval()
    
    # 3. Create Inputs with CORRECT SHAPE
    # QPOS: (Batch, Dim) -> (1, 14)
    dummy_qpos = torch.from_numpy(stats['qpos_mean']).float().cuda().unsqueeze(0) 
    
    # Image: (Batch, Num_Cameras, Channel, Height, Width) -> (1, 1, 3, 480, 640)
    # This matches the fix I gave you for residual_env.py
    dummy_img = torch.zeros((1, 1, 3, 480, 640)).float().cuda()
    
    # 4. Run Inference
    with torch.inference_mode():
        # Policy expects normalized qpos (which dummy_qpos is, since we used mean)
        normalized_qpos = torch.zeros_like(dummy_qpos)
        
        output = policy(normalized_qpos, dummy_img)
        raw_action = output[:, 0, :].cpu().numpy().flatten()
        
        print(f"\nRaw Policy Output (Normalized): {raw_action[:5]}")
        
        # Un-normalize
        real_action = raw_action * stats['action_std'] + stats['action_mean']
        print(f"Un-normalized Command:        {real_action[:5]}")
        
    print("\nCheck: Are 'Un-normalized Command' values small numbers (e.g. -3.14 to 3.14)?")

if __name__ == "__main__":
    ckpt = 'src/demospeedup/aloha/data/outputs/ACT_ckpt/sim_transfer_cube_scripted'
    debug(ckpt)