import os
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
import h5py
import numpy as np

# --- CONFIGURATION ---
# Base path to bags - change according to user environment
BASE_DIR = '/home/seungsuplee/vscode/projects/2025-2-CS672/src/sidetracks/src/replayer/scripts/bags'
OUTPUT_HDF5 = 'sidetracks_dataset.hdf5'
JOINT_TOPIC = '/my_gen3_lite/joint_states'
NUM_USERS = 40  # user_0 to user_39

def process_single_bag(bag_path):
    """Extracts joint positions from a single bag file."""
    joint_positions = []
    
    try:
        with Reader(bag_path) as reader:
            # Filter for the connection to our specific topic
            connections = [x for x in reader.connections if x.topic == JOINT_TOPIC]
            
            if not connections:
                print(f"  [WARN] Topic {JOINT_TOPIC} not found in {bag_path}")
                return None

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                joint_positions.append(msg.position)
                
    except Exception as e:
        print(f"  [ERROR] Failed to read {bag_path}: {e}")
        return None

    if not joint_positions:
        return None
        
    return np.array(joint_positions)

def main():
    # Create the HDF5 file
    with h5py.File(OUTPUT_HDF5, 'w') as f:
        data_group = f.create_group("data")
        
        success_count = 0
        
        for i in range(NUM_USERS):
            user_folder = f"user_{i}"
            bag_file = "0.bag"
            bag_path = os.path.join(BASE_DIR, user_folder, bag_file)
            
            if not os.path.exists(bag_path):
                print(f"Skipping {user_folder}: {bag_path} does not exist.")
                continue
                
            print(f"Processing {user_folder}...")
            
            # Extract data
            qpos_data = process_single_bag(bag_path)
            
            if qpos_data is not None:
                # Create a group for this demo (e.g., data/demo_0, data/demo_1...)
                # We map user_0 -> demo_0, user_1 -> demo_1, etc.
                demo_grp = data_group.create_group(f"demo_{i}")
                
                # Save joint positions
                # Dataset path: /data/demo_X/qpos
                demo_grp.create_dataset('qpos', data=qpos_data)
                
                # Metadata
                demo_grp.attrs['num_frames'] = len(qpos_data)
                demo_grp.attrs['user_id'] = user_folder
                
                print(f"  -> Saved {len(qpos_data)} frames.")
                success_count += 1
            else:
                print(f"  -> No data extracted.")

    print(f"\nBatch processing complete. Successfully saved {success_count} demos to {OUTPUT_HDF5}")

if __name__ == "__main__":
    main()