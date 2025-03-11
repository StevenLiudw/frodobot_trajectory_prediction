import os
import shutil
import sys
import pickle as pkl
import argparse
from tqdm import tqdm
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path to import from wp.data_utils.dataloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wp.data_utils.dataloader import TrajectoryDataset

def check_labels(data_dir, output_dir=None, verbose=False, visualize=False):
    """
    Check all existing labels in the dataset and print statistics.
    
    Args:
        data_dir (str): Path to the data directory
        output_dir (str, optional): Directory to save statistics and visualizations
        verbose (bool): Whether to print detailed information for each label
        visualize (bool): Whether to visualize samples with invalid, collision, or off-road labels
    
    Returns:
        dict: Statistics about the labels
    """
    
    # Initialize dataset to get sample information
    dataset = TrajectoryDataset(data_dir=data_dir, n_waypoints=10,dummy_goal=True)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if visualize:
            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
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
                        is_valid = label.get("valid", False)
                        has_collision = label.get("collision", False)
                        is_off_road = label.get("off_road", False)
                        
                        if is_valid:
                            valid_count += 1
                        if has_collision:
                            collision_count += 1
                        if is_off_road:
                            off_road_count += 1
                        
                        # Store label for analysis
                        label_info = {
                            "sample_idx": i,
                            "ride_dir": ride_dir,
                            "img_idx": img_idx,
                            "valid": is_valid,
                            "collision": has_collision,
                            "off_road": is_off_road,
                            "explanation": label.get("explanation", "")
                        }
                        all_labels.append(label_info)
                        
                        # Visualize if requested and the label is interesting (invalid, collision, or off-road)
                        if visualize and output_dir and (not is_valid or has_collision or is_off_road):
                            visualize_sample(dataset, i, label, os.path.join(output_dir, "visualizations"))
                        
                        if verbose:
                            print(f"Sample {i}:")
                            print(f"  Valid: {is_valid}")
                            print(f"  Collision: {has_collision}")
                            print(f"  Off-road: {is_off_road}")
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

def prepare_image_with_waypoints(image, waypoints):
    """
    Prepare an image with projected waypoints for visualization.
    
    Args:
        image (np.ndarray): The original image
        waypoints (np.ndarray): The waypoints to project
        
    Returns:
        np.ndarray: Image with projected waypoints
    """
    # Camera parameters (same as in auto_label_async.py)
    camera_height = 0.561  # meters above ground
    K = np.array([
        [203.93, 0, 192], 
        [0, 203.933, 144], 
        [0, 0, 1]
    ])
    dist_coeffs = np.array([-0.2172, 0.0537, 0.001853, -0.002105, -0.006000])
    
    # Create a copy for drawing
    image_with_waypoints = image.copy()
    
    # Convert waypoints to camera coordinates (vectorized)
    waypoints_camera = np.column_stack([
        -waypoints[:, 1],                  # x: -y_world
        np.full(len(waypoints), camera_height),  # y: camera_height
        waypoints[:, 0]                    # z: x_world
    ])
    
    # Filter out points behind the camera
    valid_indices = waypoints_camera[:, 2] > 0
    valid_waypoints_camera = waypoints_camera[valid_indices]
    
    # If no valid waypoints, return early
    if len(valid_waypoints_camera) == 0:
        return image_with_waypoints
    
    # Project to normalized image coordinates (before distortion)
    normalized_points = valid_waypoints_camera[:, :2] / valid_waypoints_camera[:, 2:3]
    
    # Apply distortion to normalized points
    x = normalized_points[:, 0]
    y = normalized_points[:, 1]
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r2*r4
    
    # Radial distortion
    k1, k2, p1, p2, k3 = dist_coeffs
    radial_distortion = 1 + k1*r2 + k2*r4 + k3*r6
    
    # Tangential distortion
    dx = 2*p1*x*y + p2*(r2 + 2*x*x)
    dy = p1*(r2 + 2*y*y) + 2*p2*x*y
    
    # Apply distortion
    distorted_x = x*radial_distortion + dx
    distorted_y = y*radial_distortion + dy
    
    # Stack distorted coordinates
    distorted_points = np.column_stack([distorted_x, distorted_y, np.ones(len(distorted_x))])
    
    # Apply camera intrinsics to get pixel coordinates
    waypoints_image = (K @ distorted_points.T).T
    
    # Convert to integer coordinates
    waypoints_image_int = waypoints_image[:, :2].astype(int)
    
    # Check which points are within image bounds
    in_bounds = (
        (0 <= waypoints_image_int[:, 0]) & 
        (waypoints_image_int[:, 0] < image_with_waypoints.shape[1]) &
        (0 <= waypoints_image_int[:, 1]) & 
        (waypoints_image_int[:, 1] < image_with_waypoints.shape[0])
    )
    
    # Draw only the in-bounds waypoints
    visible_indices = np.where(in_bounds)[0]
    for idx in visible_indices:
        x, y = waypoints_image_int[idx]
        
        # Calculate original waypoint index for color
        original_idx = np.where(valid_indices)[0][idx]
        color_factor = original_idx / len(waypoints)
        
        color = (
            int(255 * (1 - color_factor)),  # R
            int(255 * color_factor),        # G
            0                               # B
        )
        cv2.circle(image_with_waypoints, (x, y), 5, color, -1)
    
    return image_with_waypoints

def visualize_sample(dataset, sample_idx, label, output_dir):
    """
    Visualize a sample with its waypoints and label information.
    
    Args:
        dataset (TrajectoryDataset): The dataset
        sample_idx (int): The index of the sample
        label (dict): The label information
        output_dir (str): Directory to save the visualization
    """
    try:
        # Get the sample
        sample = dataset[sample_idx]
        
        # Get image and waypoints
        image = sample['image']
        waypoints = sample['waypoints']
        
        # Convert image tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            if image.max() <= 1.0:
                image = image.permute(1, 2, 0).numpy()
            else:
                image = image.permute(1, 2, 0).numpy() / 255.0
            image = (image * 255).astype(np.uint8)
        
        # Prepare image with waypoints
        image_with_waypoints = prepare_image_with_waypoints(image, waypoints)
        
        # Convert to PIL Image for text drawing
        pil_image = Image.fromarray(image_with_waypoints)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        # Add label information to the image
        text_lines = [
            f"Valid: {label.get('valid', False)}",
            f"Collision: {label.get('collision', False)}",
            f"Off-Road: {label.get('off_road', False)}",
        ]
        
        # Draw text with background
        y_position = 10
        for line in text_lines:
            text_width, text_height = draw.textsize(line, font=font) if hasattr(draw, 'textsize') else (100, 20)
            draw.rectangle([(10, y_position), (10 + text_width, y_position + text_height)], fill=(0, 0, 0, 128))
            draw.text((10, y_position), line, fill=(255, 255, 255), font=font)
            y_position += text_height + 5
        
        # # Add explanation if available
        # explanation = label.get("explanation", "")
        # if explanation:
        #     # Wrap explanation text to fit in the image
        #     max_width = pil_image.width - 20
        #     wrapped_text = []
        #     current_line = ""
            
        #     for word in explanation.split():
        #         test_line = current_line + " " + word if current_line else word
        #         text_width, _ = draw.textsize(test_line, font=font) if hasattr(draw, 'textsize') else (len(test_line) * 8, 20)
                
        #         if text_width <= max_width:
        #             current_line = test_line
        #         else:
        #             wrapped_text.append(current_line)
        #             current_line = word
            
        #     if current_line:
        #         wrapped_text.append(current_line)
            
        #     # Draw explanation text
        #     for line in wrapped_text:
        #         text_width, text_height = draw.textsize(line, font=font) if hasattr(draw, 'textsize') else (len(line) * 8, 20)
        #         draw.rectangle([(10, y_position), (10 + text_width, y_position + text_height)], fill=(0, 0, 0, 128))
        #         draw.text((10, y_position), line, fill=(255, 255, 255), font=font)
        #         y_position += text_height + 5
        
        # Save the image
        label_type = []
        if not label.get('valid', False):
            label_type.append("invalid")
        if label.get('collision', False):
            label_type.append("collision")
        if label.get('off_road', False):
            label_type.append("off_road")
        
        label_type_str = "_".join(label_type) if label_type else "unknown"
        
        # Get ride_dir and img_idx from the dataset
        ride_dir = dataset.samples[sample_idx]["ride_dir"]
        img_idx = dataset.samples[sample_idx]["img_idx"]
        
        # Create a filename that includes the ride and image info
        ride_name = os.path.basename(ride_dir)
        filename = f"{sample_idx}_{ride_name}_{img_idx}_{label_type_str}.jpg"
        
        pil_image.save(os.path.join(output_dir, filename))
        
    except Exception as e:
        print(f"Error visualizing sample {sample_idx}: {e}")

def remove_all_labels(data_dir, verbose=False):
    """
    Remove all existing labels in the dataset.
    
    Args:
        data_dir (str): Path to the data directory
        verbose (bool): Whether to print detailed information for each label
    
    Returns:
        dict: Statistics about the removed labels
    """
    
    confirm = input("Are you sure you want to remove all labels? (y/n): ")
    if confirm != "y":
        print("Exiting...")
        return

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
    parser.add_argument("--output_dir", type=str, default="out/check_labels", help="Directory to save statistics and visualizations")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information for each label")
    parser.add_argument("--remove_labels", action="store_true", help="Remove all existing labels")
    parser.add_argument("--visualize", action="store_true", help="Visualize samples with invalid, collision, or off-road labels")
    
    args = parser.parse_args()
    
    if args.remove_labels:
        # Remove labels
        remove_all_labels(args.data_dir, args.verbose)
    else:
        # Check labels
        check_labels(args.data_dir, args.output_dir, args.verbose, args.visualize)

if __name__ == "__main__":
    main()
