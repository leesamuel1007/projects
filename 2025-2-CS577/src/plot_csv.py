import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Update these paths to match your actual CSV filenames
files = {
    "RL-All": "results/residual_all_PPO_1.csv",   # Replace with your filename
    "RL-High": "results/residual_high_PPO_1.csv", # Replace with your filename
    "RL-Low": "results/residual_low_PPO_1.csv"    # Replace with your filename
}

# Plot settings
max_rows = 70  # Limit data to first 200k rows
output_file = "results/episode_reward_comparison.png"

def plot_rewards():
    plt.figure(figsize=(12, 6))
    
    # Colors for consistency with your previous plots
    colors = {'RL-All': 'orange', 'RL-High': 'red', 'RL-Low': 'blue'}
    
    for label, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found. Skipping.")
            continue
            
        print(f"Processing {label} from {filepath}...")
        
        # 1. Read CSV
        # 'header=0' infers headers from the first row
        df = pd.read_csv(filepath)
        
        # 2. Extract Data
        # We want the 3rd column (index 2) named "Value"
        # We also limit to max_rows
        try:
            # Check if 'Value' exists by name, otherwise use index
            if 'Value' in df.columns:
                data = df['Value']
            else:
                data = df.iloc[:, 2] # Fallback to 3rd column by index
            
            # Slice: from start to 140k
            data = data.iloc[:max_rows]
            
            # 3. Create X-axis (Shifted Row Number)
            # 0, 1, 2, ... N
            x_axis = range(len(data))
            
            # 4. Plot
            plt.plot(x_axis, data, 
                     label=label, 
                     color=colors.get(label, 'gray'), 
                     alpha=0.8, 
                     linewidth=1.5)
            
        except Exception as e:
            print(f"Error processing {label}: {e}")

    # Formatting
    plt.title(f"Episode Reward Mean")
    plt.xlabel("Training Step (x1024)")
    plt.ylabel("Episode Reward Mean")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_rewards()