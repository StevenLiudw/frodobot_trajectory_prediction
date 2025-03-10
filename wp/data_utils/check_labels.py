import os
import shutil
import sys
import pickle as pkl
import argparse
from tqdm import tqdm
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Add parent directory to path to import from wp.data_utils.dataloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wp.data_utils.dataloader import TrajectoryDataset

def check_labels(data_dir, output_dir=None, verbose=False):
    """
    Check all existing labels in the dataset and print statistics.
    
    Args:
        data_dir (str): Path to the data directory
        output_dir (str, optional): Directory to save statistics and visualizations
        verbose (bool): Whether to print detailed information for each label
    
    Returns:
        dict: Statistics about the labels
    """
    confirm = input("Are you sure you want to check labels? (y/n): ")
    if confirm != "y":
        print("Exiting...")
        return
    
    # Initialize dataset to get sample information
    dataset = TrajectoryDataset(data_dir=data_dir, n_waypoints=10)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counters and storage
    total_samples = len(dataset.samples)
    labeled_count = 0
    insufficient_waypoints_count = 0
    valid_count = 0
    collision_count = 0
    off_road_count = 0
    
    # Store all labels for analysis
    all_labels = []
    
    print(f"Checking labels for {total_samples} samples in {data_dir}")
    
    # Iterate through all samples
    for i, sample_info in enumerate(tqdm(dataset.samples)):
        ride_dir = sample_info["ride_dir"]
        img_idx = sample_info["img_idx"]
        
        # Path to the label file
        label_path = os.path.join(ride_dir, "labels", f"{img_idx}.pkl")
        
        # Check if label exists
        if os.path.exists(label_path):
            labeled_count += 1
            
            # Load the label
            with open(label_path, "rb") as f:
                try:
                    label = pkl.load(f)
                    
                    # Check if it's an "insufficient waypoints" label
                    if label.get("insufficient_waypoints", False):
                        insufficient_waypoints_count += 1
                        if verbose:
                            print(f"Sample {i}: Insufficient waypoints")
                    else:
                        # Count label statistics
                        if label.get("valid", False):
                            valid_count += 1
                        if label.get("collision", False):
                            collision_count += 1
                        if label.get("off_road", False):
                            off_road_count += 1
                        
                        # Store label for analysis
                        label_info = {
                            "sample_idx": i,
                            "ride_dir": ride_dir,
                            "img_idx": img_idx,
                            "valid": label.get("valid", False),
                            "collision": label.get("collision", False),
                            "off_road": label.get("off_road", False),
                            "explanation": label.get("explanation", "")
                        }
                        all_labels.append(label_info)
                        
                        if verbose:
                            print(f"Sample {i}:")
                            print(f"  Valid: {label.get('valid', False)}")
                            print(f"  Collision: {label.get('collision', False)}")
                            print(f"  Off-road: {label.get('off_road', False)}")
                            print(f"  Explanation: {label.get('explanation', '')}")
                            print()
                except Exception as e:
                    print(f"Error loading label for sample {i}: {e}")
    
    # Calculate statistics
    unlabeled_count = total_samples - labeled_count
    labeled_percent = (labeled_count / total_samples) * 100 if total_samples > 0 else 0
    valid_percent = (valid_count / (labeled_count - insufficient_waypoints_count)) * 100 if (labeled_count - insufficient_waypoints_count) > 0 else 0
    collision_percent = (collision_count / (labeled_count - insufficient_waypoints_count)) * 100 if (labeled_count - insufficient_waypoints_count) > 0 else 0
    off_road_percent = (off_road_count / (labeled_count - insufficient_waypoints_count)) * 100 if (labeled_count - insufficient_waypoints_count) > 0 else 0
    
    # Print summary statistics
    print("\nLabel Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Labeled samples: {labeled_count} ({labeled_percent:.2f}%)")
    print(f"Unlabeled samples: {unlabeled_count} ({100 - labeled_percent:.2f}%)")
    print(f"Insufficient waypoints: {insufficient_waypoints_count} ({(insufficient_waypoints_count / labeled_count) * 100:.2f}% of labeled)")
    print(f"Valid paths: {valid_count} ({valid_percent:.2f}% of labeled with sufficient waypoints)")
    print(f"Collision paths: {collision_count} ({collision_percent:.2f}% of labeled with sufficient waypoints)")
    print(f"Off-road paths: {off_road_count} ({off_road_percent:.2f}% of labeled with sufficient waypoints)")
    
    # Create a DataFrame for more analysis if we have labels
    if all_labels:
        df = pd.DataFrame(all_labels)
        
        # Count combinations of labels
        label_combinations = Counter(zip(df['valid'], df['collision'], df['off_road']))
        
        print("\nLabel Combinations (valid, collision, off_road):")
        for (valid, collision, off_road), count in label_combinations.items():
            print(f"  {valid}, {collision}, {off_road}: {count} ({(count / len(df)) * 100:.2f}%)")
        
        # Save to CSV if output directory is specified
        if output_dir:
            df.to_csv(os.path.join(output_dir, "label_analysis.csv"), index=False)
            
            # Create some visualizations
            plt.figure(figsize=(10, 6))
            plt.bar(['Valid', 'Collision', 'Off-road'], 
                    [valid_count, collision_count, off_road_count])
            plt.title('Label Distribution')
            plt.ylabel('Count')
            plt.savefig(os.path.join(output_dir, "label_distribution.png"))
            
            # Combination chart
            labels = [f"{v}, {c}, {o}" for (v, c, o) in label_combinations.keys()]
            counts = list(label_combinations.values())
            
            plt.figure(figsize=(12, 6))
            plt.bar(labels, counts)
            plt.title('Label Combinations (valid, collision, off-road)')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "label_combinations.png"))
    
    # Return statistics
    return {
        "total_samples": total_samples,
        "labeled_count": labeled_count,
        "unlabeled_count": unlabeled_count,
        "insufficient_waypoints_count": insufficient_waypoints_count,
        "valid_count": valid_count,
        "collision_count": collision_count,
        "off_road_count": off_road_count
    }

def remove_all_labels(data_dir, verbose=False):
    """
    Remove all existing labels in the dataset.
    
    Args:
        data_dir (str): Path to the data directory
        verbose (bool): Whether to print detailed information for each label
    
    Returns:
        dict: Statistics about the removed labels
    """
    # Initialize dataset to get sample information
    dataset = TrajectoryDataset(data_dir=data_dir, n_waypoints=10)
    
    # Initialize counters
    total_samples = len(dataset.samples)
    removed_count = 0
    
    print(f"Scanning for labels in {total_samples} samples in {data_dir}")
    
    # Iterate through all samples
    for i, sample_info in enumerate(tqdm(dataset.samples)):
        ride_dir = sample_info["ride_dir"]
        label_dir = os.path.join(ride_dir, "labels")
        if os.path.exists(label_dir):
            shutil.rmtree(label_dir, ignore_errors=True)
            removed_count += 1
            if verbose:
                print(f"Removed label: {label_dir}")
    
    # Print summary statistics
    print("\nLabel Removal Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Labels found: {removed_count}")
    
    # Return statistics
    return {
        "total_samples": total_samples,
        "labels_found": removed_count,
        "labels_removed": removed_count,
    }

def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="Check or remove waypoint labels")
    parser.add_argument("--data_dir", type=str, default="data/filtered_2k", help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save statistics and visualizations")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information for each label")
    parser.add_argument("--remove_labels", action="store_true", help="Remove all existing labels")
    
    args = parser.parse_args()
    
    if args.remove_labels:
        # Remove labels
        remove_all_labels(args.data_dir, args.verbose)
    else:
        # Check labels
        check_labels(args.data_dir, args.output_dir, args.verbose)

if __name__ == "__main__":
    main()
