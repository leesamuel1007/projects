import os
import argparse
import pickle
import numpy as np
import imageio
import pandas as pd
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from stable_baselines3 import PPO

sys.path.append(os.getcwd())
from src.residual_env import ResidualAlohaEnv

def plot_colored_3d_trajectory(all_trajectories, all_magnitudes, mode):

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize magnitude for colormap (0 to approx max observed, e.g., 1.0)
    norm = plt.Normalize(vmin=0.0, vmax=0.8) 
    cmap = plt.get_cmap('coolwarm') # Blue (Low) -> Red (High)

    print(f"  Plotting 3D trajectories for {mode}...")
    
    for traj in all_trajectories:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='gray', alpha=0.1, linewidth=0.5)

    for i, (traj, mags) in enumerate(zip(all_trajectories, all_magnitudes)):
        # Create segments for Line3DCollection
        points = traj[:, :3].reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        seg_colors = cmap(norm(mags[:len(segments)]))
        
        lc = Line3DCollection(segments, colors=seg_colors, alpha=0.7, linewidth=1.5)
        ax.add_collection3d(lc)
        
        if i == 0:
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=20, label='Start')
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='black', s=20, label='End')
        else:
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=10)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='black', s=10)

    # Auto-scale axes
    all_points = np.vstack(all_trajectories)
    ax.set_xlim(all_points[:,0].min(), all_points[:,0].max())
    ax.set_ylim(all_points[:,1].min(), all_points[:,1].max())
    ax.set_zlim(all_points[:,2].min(), all_points[:,2].max())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Trajectory Intervention Map: {mode.upper()}\n(Blue=Copy Base, Red=Edit/Residual)')
    
    # Add Colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(all_magnitudes[0]) # Dummy array
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Residual Action Magnitude ||a_res||')
    
    plt.legend()
    save_path = f"results/3d_intervention_{mode}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved 3D intervention plot to {save_path}")

def evaluate(args):
    # Load Zones
    with open(args.zones_file, 'rb') as f:
        zones = pickle.load(f)
    
    modes = ['baseline', 'high', 'low', 'all']
    final_results = []
    
    for mode in modes:
        print(f"\n>>> EVALUATING MODE: {mode.upper()}")
        
        # 1. Setup Active Steps & Model Path
        if mode == 'baseline':
            active_steps = []
            model_path = None
        elif mode == 'all':
            active_steps = list(range(1000))
            model_path = "checkpoints/residual_all/final_model"
        else:
            active_steps = zones[mode]
            model_path = f"checkpoints/residual_{mode}/final_model"

        # 2. Load Residual Model (Skip for Baseline)
        model = None
        if mode != 'baseline':
            if not os.path.exists(model_path + ".zip"):
                print(f"  [Warn] Model {model_path} not found. Skipping.")
                continue
            model = PPO.load(model_path)

        # 3. Init Env
        env = ResidualAlohaEnv(
            task_name=args.task_name,
            ckpt_dir=args.base_policy_ckpt,
            active_steps=active_steps
        )

        success_count = 0
        rewards = []
        
        all_magnitudes = [] # Store residual norms for every episode
        all_trajectories = [] 
        
        # Video Writer
        os.makedirs("results", exist_ok=True)
        video_path = f"results/video_{mode}.mp4"
        writer = imageio.get_writer(video_path, fps=30)
        
        for i in range(args.num_rollouts):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            episode_mags = []
            episode_traj = [] 
            
            while not done:
                # A. Capture Video (First episode only)
                if i == 0:
                    img = env.env._physics.render(height=480, width=640, camera_id='top')
                    writer.append_data(img)
                
                # B. Record Trajectory (EE Position)
                qpos = env.env._physics.data.qpos
                episode_traj.append(qpos[:3].copy()) 

                # C. Get Residual Action & Magnitude
                if mode == 'baseline':
                    action = np.zeros(14) 
                    res_mag = 0.0
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    res_mag = np.linalg.norm(action)
                
                episode_mags.append(res_mag)

                # D. Step
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            rewards.append(total_reward)
            all_magnitudes.append(np.array(episode_mags))
            all_trajectories.append(np.array(episode_traj))
            
            if total_reward >= 4: success_count += 1
            print(f"  Trial {i+1}: Reward={total_reward:.1f} | Avg Mag={np.mean(episode_mags):.4f}")

        writer.close()
        
        # --- PLOT 3D TRAJECTORY ---
        plot_colored_3d_trajectory(all_trajectories, all_magnitudes, mode)

        # --- METRICS ---
        # Calculate mean intervention for CSV
        avg_intervention = np.mean([np.mean(m) for m in all_magnitudes])
        success_rate = (success_count / args.num_rollouts) * 100
        
        final_results.append({
            "Mode": mode,
            "Success Rate (%)": success_rate,
            "Avg Return": np.mean(rewards),
            "Avg Intervention": avg_intervention
        })
        env.close()

    # Save Final CSV
    df = pd.DataFrame(final_results)
    df.to_csv("results/final_comparison.csv", index=False)
    
    print("\nEvaluation Complete. Summary:")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='sim_transfer_cube_scripted')
    parser.add_argument('--base_policy_ckpt', default='src/demospeedup/aloha/data/outputs/ACT_ckpt/sim_transfer_cube_scripted')
    parser.add_argument('--zones_file', default='data/analysis/training_zones.pkl')
    parser.add_argument('--num_rollouts', type=int, default=20)
    args = parser.parse_args()
    
    evaluate(args)