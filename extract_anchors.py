import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
import glob
import argparse
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Extract ego anchors from .npz files using K-Means")
    parser.add_argument("--data_dir", type=str, default="/mnt/workspace/shared/pnc/data/nuplan/data_for_diffusion", help="Path to .npz files")
    parser.add_argument("--save_path", type=str, default="./ego_anchors_20.npy", help="Path to save the generated anchors")
    parser.add_argument("--num_anchors", type=int, default=20, help="Number of clusters (anchors) to generate")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum number of samples to use for clustering to avoid memory issues")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Searching for .npz files in {args.data_dir}...")
    npz_files = glob.glob(os.path.join(args.data_dir, "*.npz"))
    print(f"Found {len(npz_files)} .npz files.")

    if len(npz_files) == 0:
        print("No .npz files found. Please check your data directory.")
        return

    # To avoid loading all files if they are too many and exceeding memory
    if len(npz_files) > args.max_samples:
        print(f"Randomly sampling {args.max_samples} files from {len(npz_files)} total files...")
        np.random.seed(args.seed)
        npz_files = np.random.choice(npz_files, args.max_samples, replace=False).tolist()

    trajectories = []
    
    print("Loading ego trajectories...")
    for file in tqdm(npz_files):
        try:
            data = np.load(file)
            if "ego_agent_future" in data:
                ego_future = data["ego_agent_future"] # Shape typically [80, 3] (x, y, heading) relative to current ego
                trajectories.append(ego_future)
            else:
                print(f"Warning: 'ego_agent_future' not found in {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    if not trajectories:
        print("No valid trajectories collected. Exiting.")
        return

    trajectories = np.stack(trajectories) # [N, T, 3]
    N, T, D = trajectories.shape
    print(f"Successfully collected {N} trajectories of shape [{T}, {D}]")

    print(f"Running K-Means clustering to generate {args.num_anchors} anchors...")
    # Flatten the temporal and feature dimensions: [N, T, D] -> [N, T * D]
    trajectories_flat = trajectories.reshape(N, -1)
    
    # Run K-Means
    kmeans = KMeans(n_clusters=args.num_anchors, random_state=args.seed, n_init="auto")
    kmeans.fit(trajectories_flat)
    
    # Reshape the cluster centers back to [K, T, D]
    anchors_flat = kmeans.cluster_centers_ 
    anchors = anchors_flat.reshape(args.num_anchors, T, D)

    print(f"Saving generated anchors to {args.save_path}...")
    np.save(args.save_path, anchors)
    print("Done! You can use this file for training with:")
    print(f"  --use_ego_anchor true --ego_anchor_path {os.path.abspath(args.save_path)} --ego_anchor_state_format absolute --num_ego_anchors {args.num_anchors}")

if __name__ == "__main__":
    main()
