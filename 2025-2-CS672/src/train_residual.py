import os
import argparse
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import sys

# Import wrapper
sys.path.append(os.getcwd())
from src.residual_env import ResidualAlohaEnv

def train(args):
    # Load Active Steps
    if args.mode == 'all':
        active_steps = list(range(1000))
    else:
        with open(args.zones_file, 'rb') as f:
            zones = pickle.load(f)
        active_steps = zones[args.mode] 

    print(f"Training Mode: {args.mode.upper()} | Active Steps: {len(active_steps)}")

    env = ResidualAlohaEnv(
        task_name=args.task_name,
        ckpt_dir=args.base_policy_ckpt,
        active_steps=active_steps
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=f"logs/residual_{args.mode}"
    )

    # Save checkpoints every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=f"checkpoints/residual_{args.mode}/",
        name_prefix="res_model"
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    model.save(f"checkpoints/residual_{args.mode}/final_model")
    print(f"Finished {args.mode}. Saved to checkpoints/residual_{args.mode}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # UPDATED DEFAULT PATH
    parser.add_argument('--base_policy_ckpt', default='src/demospeedup/aloha/data/outputs/ACT_ckpt/sim_transfer_cube_scripted')
    parser.add_argument('--zones_file', default='data/analysis/training_zones.pkl')
    parser.add_argument('--mode', choices=['high', 'low', 'all'], required=True)
    parser.add_argument('--timesteps', type=int, default=200000) # Reduced to 200k for speed
    parser.add_argument('--task_name', default='sim_transfer_cube_scripted')
    args = parser.parse_args()
    
    train(args)