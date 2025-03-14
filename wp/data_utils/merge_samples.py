import os
import shutil
from pathlib import Path
import glob
import argparse

def merge_samples(source_dir, target_dir):
    """
    Merge all .jpg files from subset_* directories within sampled_frames_* directories
    into a single directory called sampled_frames_all.
    
    Args:
        source_dir: Root directory containing sampled_frames_* directories
        target_dir: Target directory to merge all .jpg files into
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all sampled_frames_* directories
    sampled_frames_dirs = glob.glob(os.path.join(source_dir, "sampled_frames_*"))
    print(f"Found {len(sampled_frames_dirs)} sampled_frames_* directories")
    # Track statistics
    total_files = 0
    
    for frames_dir in sampled_frames_dirs:
        # Find all subset_* directories within each sampled_frames_* directory
        subset_dirs = glob.glob(os.path.join(frames_dir, "subset_*"))
        print(f"Processing {frames_dir} with {len(subset_dirs)} subset directories")
        
        for subset_dir in subset_dirs:
            # Find all .jpg files in the subset directory
            jpg_files = glob.glob(os.path.join(subset_dir, "*.jpg"))
            
            # Copy each .jpg file to the target directory
            for jpg_file in jpg_files:
                filename = os.path.basename(jpg_file)
                
                # Copy the file to the target directory with the unique name
                shutil.copy2(jpg_file, os.path.join(target_dir, filename))
                total_files += 1
    
    print(f"Merged {total_files} .jpg files into {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge .jpg files from subset_* directories within sampled_frames_* directories")
    parser.add_argument("--source", type=str, default=".", help="Root directory containing sampled_frames_* directories")
    parser.add_argument("--target", type=str, default="sampled_frames_all", help="Target directory to merge all .jpg files into")
    
    args = parser.parse_args()
    
    merge_samples(args.source, args.target)
