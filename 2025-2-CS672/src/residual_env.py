import gymnasium as gym
import numpy as np
import torch
import os
import sys
import pickle
from stable_baselines3 import PPO

class ResidualAlohaEnv(gym.Env):
    def __init__(self, task_name, ckpt_dir, active_steps=None, max_steps=400):
        super().__init__()
        self.task_name = task_name
        
        # --- 1. SETUP PATHS ---
        aloha_path = os.path.join(os.getcwd(), 'src/demospeedup/aloha')
        act_path = os.path.join(os.getcwd(), 'src/demospeedup/aloha/act')
        if aloha_path not in sys.path:
            sys.path.append(aloha_path)
        if act_path not in sys.path:
            sys.path.append(act_path)

        # --- 2. START ARGUMENT PATCH ---
        self._real_argv = sys.argv.copy()
        sys.argv = [
            'residual_env.py',          
            '--ckpt_dir', 'dummy',      
            '--policy_class', 'ACT',    
            '--task_name', 'dummy',     
            '--seed', '0',              
            '--num_epochs', '1'         
        ]

        try:
            # Import ACT classes
            from act.policy import ACTPolicy
            from act.sim_env import make_sim_env, BOX_POSE
            from act.act_utils import sample_box_pose
            
            # Store refs for use in methods
            self.PolicyClass = ACTPolicy
            self.make_sim_env = make_sim_env
            self.BOX_POSE = BOX_POSE
            self.sample_box_pose = sample_box_pose

            # --- 3. INITIALIZE ENVIRONMENT ---
            sim_task = 'sim_transfer_cube_scripted' if 'transfer_cube' in task_name else task_name
            self.env = self.make_sim_env(sim_task)
            self.max_steps = max_steps

            # Active Zones
            self.active_steps = set(active_steps) if active_steps is not None else set(range(1000))

            # Spaces
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-0.05, high=0.05, shape=(14,), dtype=np.float32
            )

            # --- 4. LOAD POLICY ---
            self.policy, self.stats = self._load_policy(ckpt_dir)
            self.time_step = 0
            self._last_obs = None # Cache for the observation
            
        except ImportError as e:
            print("Failed to import ACT. Check your paths!")
            raise e
        finally:
            sys.argv = self._real_argv

    def _load_policy(self, ckpt_dir):
        # Load Stats
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats not found at {stats_path}")

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
            
        policy_config = {
            "lr": 1e-5, "num_queries": 50, "kl_weight": 10, 
            "hidden_dim": 512, "dim_feedforward": 3200,
            "lr_backbone": 1e-5, "backbone": "resnet18",
            "enc_layers": 4, "dec_layers": 7, "nheads": 8,
            "camera_names": ['top']
        }
        
        # Instantiate Policy
        policy = self.PolicyClass(policy_config)
        
        ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
            
        print(f"Loading Base Policy from {ckpt_path}...")
        policy.load_state_dict(torch.load(ckpt_path))
        policy.cuda()
        policy.eval()
        return policy, stats

    def _process_obs(self, obs):
        qpos = obs['qpos']
        qpos_norm = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        return torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)

    def _process_img(self, obs):
        # 1. Get image (480, 640, 3)
        img = obs['images']['top'] / 255.0
        
        # 2. To Tensor and Permute to (C, H, W) -> (3, 480, 640)
        img_tensor = torch.from_numpy(img).float().cuda().permute(2, 0, 1)
        
        # 3. Add Batch AND Camera dimensions -> (1, 1, 3, 480, 640)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        
        return img_tensor

    def reset(self, seed=None, options=None):
        self.time_step = 0
        
        # Initialize Box Pose
        if 'transfer_cube' in self.task_name:
             self.BOX_POSE[0] = self.sample_box_pose()
        
        ts = self.env.reset()
        self._last_obs = ts.observation # Cache the initial observation
        
        return np.array(self._last_obs['qpos'], dtype=np.float32), {}

    def step(self, residual_action):
        # Use the cached observation from the previous step/reset
        ts_obs = self._last_obs
        
        with torch.inference_mode():
            qpos_t = self._process_obs(ts_obs)
            img_t = self._process_img(ts_obs)
            
            # ACT outputs a chunk. We take index 0.
            base_action_chunk = self.policy(qpos_t, img_t)
            base_action_raw = base_action_chunk[:, 0, :] 
            
            base_action = base_action_raw.cpu().numpy().flatten()
            base_action = base_action * self.stats['action_std'] + self.stats['action_mean']

        final_action = base_action
        if self.time_step in self.active_steps:
            final_action = base_action + residual_action
        
        ts = self.env.step(final_action)
        self.time_step += 1
        
        # Update the cached observation for the next step
        self._last_obs = ts.observation #
        
        next_obs = np.array(self._last_obs['qpos'], dtype=np.float32)
        reward = ts.reward
        terminated = (reward >= self.env.task.max_reward)
        truncated = (self.time_step >= self.max_steps)
        
        return next_obs, reward, terminated, truncated, {}