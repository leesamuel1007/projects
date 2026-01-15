import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def analyze_entropy(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
    files.sort()
    
    all_entropies = []
    all_labels = []
    
    # 1. Load Data
    max_len = 0
    for f in files:
        with h5py.File(os.path.join(dataset_dir, f), 'r') as hf:
            if 'entropy' in hf:
                e = hf['entropy'][()]
                l = hf['labels'][()] if 'labels' in hf else np.zeros_like(e)
                
                all_entropies.append(e)
                all_labels.append(l)
                max_len = max(max_len, len(e))
                
    # 2. Pad Sequences
    padded_entropy = np.zeros((len(all_entropies), max_len))
    padded_labels = np.zeros((len(all_labels), max_len))
    
    for i, (e, l) in enumerate(zip(all_entropies, all_labels)):
        length = len(e)
        padded_entropy[i, :length] = e
        padded_labels[i, :length] = l
        
    avg_entropy = np.mean(padded_entropy, axis=0)
    avg_labels = np.mean(padded_labels, axis=0)
    
    # 3. Define Zones based on Average Labels
    high_entropy_mask = avg_labels > 0.5 
    
    # 4. Visualization
    plt.figure(figsize=(12, 6))
    x = np.arange(max_len)
    
    # Plot Average Entropy
    plt.plot(x, avg_entropy, label='Average Entropy', color='black', linewidth=1.5)
    
    # Highlight High Entropy Zones
    plt.fill_between(x, np.min(avg_entropy), np.max(avg_entropy), 
                     where=high_entropy_mask, color='red', alpha=0.3, label='High Entropy Zone')
    
    plt.fill_between(x, np.min(avg_entropy), np.max(avg_entropy), 
                     where=~high_entropy_mask, color='blue', alpha=0.1, label='Low Entropy Zone')

    plt.title(f'Entropy Profile (N={len(files)} Demos)')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Entropy')
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "average_entropy_profile.png")
    plt.savefig(save_path)
    print(f"Entropy visualization saved to {save_path}")
    
    # 5. Save Zone Indices for Training
    high_ent_indices = np.where(high_entropy_mask)[0].tolist()
    low_ent_indices = np.where(~high_entropy_mask)[0].tolist()
    
    zones = {
        'high': high_ent_indices,
        'low': low_ent_indices,
        'all': list(range(max_len))
    }
    
    import pickle
    with open(os.path.join(output_dir, 'training_zones.pkl'), 'wb') as f:
        pickle.dump(zones, f)
    print("Zone indices saved to training_zones.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', default='data/analysis')
    args = parser.parse_args()
    
    analyze_entropy(args.dataset_dir, args.output_dir)