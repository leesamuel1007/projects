import h5py
# Replace with one of your actual file paths
f = h5py.File("data/sim_transfer_cube_scripted/episode_0.hdf5", "r")
print("Keys:", list(f.keys()))
if "labels" in f.keys():
    print("Success! Labels found.")
    print("Label shape:", f["labels"].shape)
    print("Unique values:", set(f["labels"][()]))
else:
    print("FAILURE: 'labels' key missing.")