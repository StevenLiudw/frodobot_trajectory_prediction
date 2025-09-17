import time
import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor
import pickle
import shutil
import glob
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IMAGE_GLOBS = [
    "*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff",
    "*.JPG","*.JPEG","*.PNG","*.BMP","*.WEBP","*.TIF","*.TIFF"
]

def _dataset_ann_exists(frame_path, dataset_root, ann_name="set1_annotation"):
    imname = os.path.splitext(os.path.basename(frame_path))[0]
    return os.path.exists(os.path.join(dataset_root, ann_name, imname, "1.png"))

def list_images(dir_path):
    paths = []
    for pat in IMAGE_GLOBS:
        paths.extend(glob.glob(os.path.join(dir_path, pat)))
    # de-dup + stable order
    return sorted(set(paths))


def _safe_rmtree(path):
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"[cleanup] Deleted temp dir: {path}")
    except Exception as e:
        logger.warning(f"[cleanup] Failed to delete {path}: {e}")

def save_to_set_dirs(image_rgb, mask, frame_path, dataset_root, set_name="set1", ann_name="set1_annotation"):
    """
    Write:
      - <dataset_root>/<set_name>/<imname>/1.jpg  (RGB image)
      - <dataset_root>/<ann_name>/<imname>/1.png  (red-on-black, mask>0)
    """
    imname = os.path.splitext(os.path.basename(frame_path))[0]
    img_dir = os.path.join(dataset_root, set_name, imname)
    ann_dir = os.path.join(dataset_root, ann_name, imname)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # 1.jpg (keep original RGB; cv2 expects BGR)
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(img_dir, "1.jpg"), img_bgr)

    # 1.png (red-on-black with threshold=0 ⇒ mask>0)
    h, w = mask.shape[:2]
    m = (mask > 0)
    red = np.zeros((h, w, 3), dtype=np.uint8)
    red[m] = (0, 0, 255)  # BGR red
    cv2.imwrite(os.path.join(ann_dir, "1.png"), red)


def load_sampled_frame_and_waypoints(frame_path):
    """
    Load a sampled frame and its corresponding waypoints.
    
    Args:
        frame_path (str): Path to the sampled frame image
        
    Returns:
        tuple: (image, waypoints) where image is a numpy array and waypoints is a numpy array of shape (N, 2)
    """
    # Load the image
    image = cv2.imread(frame_path)
    if image is None:
        raise ValueError(f"Failed to load image: {frame_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Construct the path to the waypoints file
    base_path = os.path.splitext(frame_path)[0]
    waypoints_path = f"{base_path}_waypoints.npy"
    
    # Check if waypoints file exists
    if not os.path.exists(waypoints_path):
        waypoints = None
    else:
        # Load the waypoints
        waypoints = np.load(waypoints_path)
    
    return image, waypoints

def project_waypoints_to_image(waypoints, image_shape, camera_height=0.561, camera_pitch=0):
    """
    Project waypoints from local coordinate frame to image coordinates.
    
    Args:
        waypoints (np.ndarray): Array of waypoints in local frame (N, 2)
        image_shape (tuple): Shape of the image (H, W, C)
        camera_height (float): Height of camera above ground in meters
        camera_pitch (float): Camera pitch in radians
        
    Returns:
        np.ndarray: Array of projected waypoint coordinates in image frame (N, 2)
    """
    # Camera intrinsic matrix
    K = np.array([
        [203.93, 0, 192], 
        [0, 203.933, 144], 
        [0, 0, 1]
    ])
    
    # Camera distortion parameters (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([-0.2172, 0.0537, 0.001853, -0.002105, -0.006000])
    
    # Convert waypoints to camera coordinates (vectorized)
    waypoints_camera = np.column_stack([
        -waypoints[:, 1],                  # x: -y_world
        np.full(len(waypoints), camera_height),  # y: camera_height
        waypoints[:, 0]                    # z: x_world
    ])
    
    # Filter out points behind the camera
    valid_indices = waypoints_camera[:, 2] > 0
    valid_waypoints_camera = waypoints_camera[valid_indices]
    
    # If no valid waypoints, return empty array
    if len(valid_waypoints_camera) == 0:
        return np.array([])
    
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
    
    # Convert to integer coordinates - FIX: first get the float coordinates, then convert to int
    waypoints_image_float = waypoints_image[:, :2]
    waypoints_image_int = waypoints_image_float.astype(int)
    
    # Check which points are within image bounds
    in_bounds = (
        (0 <= waypoints_image_int[:, 0]) & 
        (waypoints_image_int[:, 0] < image_shape[1]) &
        (0 <= waypoints_image_int[:, 1]) & 
        (waypoints_image_int[:, 1] < image_shape[0])
    )
    
    # Return only the in-bounds waypoints
    return waypoints_image_int[in_bounds]

def segment_with_sam2(image, waypoints_2d, output_dir=None, visualize=False, predictor=None):
    """
    Use SAM2 to segment the image using projected waypoints as prompts.
    
    Args:
        image (np.ndarray): RGB image as numpy array
        waypoints_2d (np.ndarray): Projected waypoints in image coordinates (N, 2)
        output_dir (str, optional): Directory to save output visualizations
        visualize (bool): Whether to visualize the results
        predictor: Optional SAM2ImagePredictor instance (if None, one will be created)
        
    Returns:
        tuple: (masks, scores, logits) from SAM2 prediction
    """
    # Initialize SAM2 predictor if not provided
    if predictor is None:
        predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus")
    
    # Convert waypoints to point prompts format expected by SAM2
    # SAM2 expects point prompts as (N, 2) array with coordinates in (x, y) format
    point_prompts = waypoints_2d
    
    # Add label for each point (1 for foreground)
    point_labels = np.ones(len(point_prompts), dtype=int)
    
    # Prepare input prompts
    input_prompts = {
        "point_coords": point_prompts,
        "point_labels": point_labels
    }
    
    # Run inference
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(input_prompts)
    
    # Visualize if requested
    if visualize and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a figure for visualization
        plt.figure(figsize=(15, 10))
        
        # Show the original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.scatter(point_prompts[:, 0], point_prompts[:, 1], c='red', s=20)
        plt.title("Image with Waypoint Prompts")
        plt.axis('off')
        
        # Show the segmentation mask (use the highest scoring mask)
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        
        # If we have masks, overlay them
        if masks.shape[0] > 0:
            best_mask_idx = scores.argmax()
            mask = masks[best_mask_idx]
            plt.imshow(mask, alpha=0.5, cmap='jet')
            plt.title(f"Segmentation Mask (Score: {scores[best_mask_idx]:.3f})")
        else:
            plt.title("No Mask Generated")
            
        plt.axis('off')
        
        # Save the visualization
        image_name = os.path.basename(output_dir) if output_dir.endswith('.png') else 'segmentation_result.png'
        plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
        plt.close()
    
    return masks, scores, logits

def process_interactive_mode(frames_dir, output_dir, limit=None, compute_background=False, save_interactive_result=False, model_size="base", overwrite=False):
    """
    Process frames in interactive mode, allowing user to refine segmentations.
    Uses two SAM2 instances to precompute embeddings for the next image while user is working.
    
    Args:
        frames_dir (str): Directory containing sampled frames
        output_dir (str): Directory to save output segmentations
        limit (int, optional): Limit the number of frames to process
        compute_background (bool): Whether to use a second SAM2 instance for background computation
        save_interactive_result (bool): Whether to save visualization of the interactive result
        model_size (str): Size of the SAM2 model to use ("base", "small", or "tiny")
        overwrite (bool): Whether to overwrite existing segmentations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all jpg files in the directory
    frame_paths = list_images(frames_dir)
    
    if limit:
        frame_paths = frame_paths[:limit]
    
    logger.info(f"Found {len(frame_paths)} frames to process in interactive mode")
    
    # Determine model checkpoint based on size
    if model_size == "small":
        model_checkpoint = "facebook/sam2-hiera-small"
    elif model_size == "tiny":
        model_checkpoint = "facebook/sam2-hiera-tiny"
    else:  # default to base
        model_checkpoint = "facebook/sam2-hiera-base-plus"
    
    logger.info(f"Using SAM2 model: {model_checkpoint}")
    
    # Initialize SAM2 predictors - one for current frame, and optionally one for precomputing next frame
    predictor1 = SAM2ImagePredictor.from_pretrained(model_checkpoint)
    predictor2 = None
    if compute_background:
        predictor2 = SAM2ImagePredictor.from_pretrained(model_checkpoint)
    
    # Variables to store precomputed data
    next_valid_frame_index = None
    next_valid_frame_data = None
    precompute_thread = None
    
    # Process each frame
    is_first = True
    i = 0
    while i < len(frame_paths):
        try:
            frame_path = frame_paths[i]
            # Extract frame name for output
            frame_name = os.path.basename(frame_path).split('.')[0]
            frame_output_dir = os.path.join(output_dir, frame_name)
            
            # Check if this frame has already been processed
            if _dataset_ann_exists(frame_path, output_dir, ann_name="set1_annotation"):
                logger.info(f"Skipping {frame_name}: dataset mask already exists under {output_dir}/set1_annotation")
                i += 1
                continue
            
            os.makedirs(frame_output_dir, exist_ok=True)
            
            # Load current frame and waypoints
            image, waypoints = load_sampled_frame_and_waypoints(frame_path)
            
            # Project waypoints to image
            if waypoints is not None:
                projected_waypoints = project_waypoints_to_image(waypoints, image.shape)
            else:
                projected_waypoints = np.zeros((0, 2))
            
            logger.info(f"Processing {frame_path} in interactive mode")
            print(f"\nFrame: {frame_name}")
            
            # Set image for current predictor if this is the first frame or we're not using precomputed data
            if is_first or not compute_background or next_valid_frame_data is None:
                predictor1.set_image(image)
                is_first = False
            else:
                # If we have precomputed data and it matches the current frame index, use it
                if next_valid_frame_index == i:
                    # Swap predictors to use the precomputed one
                    predictor1, predictor2 = predictor2, predictor1
                    image, projected_waypoints = next_valid_frame_data
                    logger.info(f"Using precomputed embeddings for frame: {frame_path}")
                else:
                    # If precomputed data doesn't match current index, set the image normally
                    predictor1.set_image(image)
            
            # Start a new thread to find the next valid frame and precompute its embeddings
            if compute_background and predictor2 and precompute_thread is None:
                import threading
                
                def find_and_precompute_next_frame():
                    nonlocal next_valid_frame_index, next_valid_frame_data
                    time.sleep(0.3)
                    try:
                        # Start from the next frame
                        next_idx = i + 1
                        while next_idx < len(frame_paths):
                            next_frame_path = frame_paths[next_idx]
                            
                            # Skip if already processed and not overwriting
                            next_frame_name = os.path.basename(next_frame_path).split('.')[0]
                            next_frame_output_dir = os.path.join(output_dir, next_frame_name)
                            next_mask_file = os.path.join(next_frame_output_dir, "mask.npy")
                            next_label_file = os.path.join(next_frame_output_dir, "label.pkl")
                            
                            if not overwrite and (os.path.exists(next_mask_file) or os.path.exists(next_label_file)):
                                next_idx += 1
                                continue
                            
                            try:
                                # Load and check if the next frame has valid waypoints
                                next_image, next_waypoints = load_sampled_frame_and_waypoints(next_frame_path)
                                if next_waypoints is not None:
                                    next_projected_waypoints = project_waypoints_to_image(next_waypoints, next_image.shape)
                                else:
                                    next_projected_waypoints = np.zeros((0, 2))
                                
                                # Found a valid frame, precompute embeddings
                                logger.info(f"Found next valid frame: {next_frame_path}, precomputing embeddings")
                                predictor2.set_image(next_image)
                                next_valid_frame_index = next_idx
                                next_valid_frame_data = (next_image, next_projected_waypoints)
                                break
                            except Exception as e:
                                logger.error(f"Error checking frame {next_frame_path}: {e}")
                            
                            next_idx += 1
                        
                        if next_idx >= len(frame_paths):
                            logger.info("No more valid frames found for precomputation")
                            next_valid_frame_index = None
                            next_valid_frame_data = None
                    except Exception as e:
                        logger.error(f"Error in precomputation thread: {e}")
                
                precompute_thread = threading.Thread(target=find_and_precompute_next_frame)
                precompute_thread.daemon = True
                precompute_thread.start()
            
            # Run interactive segmentation with current predictor
            mask, points, labels, is_finished = interactive_segmentation_with_predictor(
                image,
                projected_waypoints,
                output_dir=frame_output_dir,
                predictor=predictor1,
                save_interactive_result=save_interactive_result,
                frame_path=frame_path,         # <-- add this
                ramp_dir="./ramp"              # <-- optional; change path if you like
            )
            
            logger.info(f"Completed interactive segmentation for {frame_path}")

            if is_finished:
                logger.info(f"User quit the interactive segmentation.")
                return
            
            # Wait for precomputation thread to finish if it's running
            if precompute_thread and precompute_thread.is_alive():
                precompute_thread.join(timeout=0.1)  # Short timeout to avoid blocking
            
            # Reset precompute thread for next iteration
            precompute_thread = None
            
            # Move to next frame
            if next_valid_frame_index is not None:
                i = next_valid_frame_index
            else:
                i += 1
            
        except Exception as e:
            logger.error(f"Error processing {frame_path} in interactive mode: {e}")
            i += 1  # Move to next frame on error

def interactive_segmentation_with_predictor(
    image,
    initial_waypoints,
    output_dir=None,
    predictor=None,
    save_interactive_result=False,
    frame_path=None,           # <-- new
    ramp_dir="./ramp"          # <-- new
):
    """
    Interactive segmentation mode that allows users to refine masks by adding points.
    Uses a provided SAM2 predictor instance.
    
    Args:
        image (np.ndarray): RGB image as numpy array
        initial_waypoints (np.ndarray): Initial waypoints in image coordinates (N, 2)
        output_dir (str, optional): Directory to save output visualizations
        predictor: SAM2ImagePredictor instance with image already set
        save_interactive_result (bool): Whether to save visualization of the interactive result
        
    Returns:
        tuple: (final_mask, all_points, all_labels, is_finished) - the final mask, points, labels and quit status
    """
    def _delete_output_dir(p):
        try:
            if p and os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
                print(f"[d] Deleted output dir: {p}")
        except Exception as e:
            print(f"[d] Failed to delete {p}: {e}")

    def _copy_to_ramp(src, dst_dir):
        try:
            if not src or not os.path.exists(src):
                print("[r] No valid source image to copy.")
                return None
            os.makedirs(dst_dir, exist_ok=True)
            base = os.path.basename(src)
            name, ext = os.path.splitext(base)
            dst = os.path.join(dst_dir, base)
            k = 1
            while os.path.exists(dst):
                dst = os.path.join(dst_dir, f"{name}_{k}{ext}")
                k += 1
            shutil.copy2(src, dst)
            print(f"[r] Copied to {dst}")
            return dst
        except Exception as e:
            print(f"[r] Failed to copy to ramp: {e}")
            return None
    def _delete_input_assets(src):
        try:
            if src and os.path.exists(src):
                os.remove(src)
                print(f"[d] Deleted source image: {src}")
            base = os.path.splitext(src)[0]
            wp = f"{base}_waypoints.npy"
            if os.path.exists(wp):
                os.remove(wp)
                print(f"[d] Deleted waypoints: {wp}")
        except Exception as e:
            print(f"[d] Failed to delete input assets: {e}")
    def _save_for_set(image_rgb, mask, frame_path,
                      dataset_root="output",
                      set_name="set1",
                      ann_name="set1_annotation"):
        """
        Save image to:  output/set1/[imagename]/1.jpg
        Save mask png to: output/set1_annotation/[imagename]/1.png
        Mask: black background with red where mask>0
        """
        imname = os.path.splitext(os.path.basename(frame_path))[0]
        img_dir = os.path.join(dataset_root, set_name, imname)
        ann_dir = os.path.join(dataset_root, ann_name, imname)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        # 1) save image as 1.jpg (RGB->BGR for cv2)
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(img_dir, "1.jpg"), img_bgr)

        # 2) save red mask as 1.png (BGR red=(0,0,255))
        h, w = mask.shape[:2]
        m = (mask > 0)  # boolean
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[m] = (0, 0, 255)
        cv2.imwrite(os.path.join(ann_dir, "1.png"), out)
        
    assert predictor is not None, "Predictor must be provided"
    
    # Current mask state
    current_mask = None
    current_score = None
    current_logits = None

    action = None

    points_file = os.path.join(output_dir, "points.npy")
    point_labels_file = os.path.join(output_dir, "point_labels.npy")
    
    valid_traj = True
    use_waypoints = True

    if os.path.exists(points_file) and os.path.exists(point_labels_file):
        print(f"Loading existing points and labels from {points_file} and {point_labels_file}")
        existing_points = np.load(points_file)
        existing_point_labels = np.load(point_labels_file)
        
        # Denormalize points
        h, w = image.shape[:2]
        existing_points[:, 0] *= float(w)
        existing_points[:, 1] *= float(h)
        existing_points = existing_points.astype(int)

        user_points = existing_points
        user_labels = existing_point_labels
        if os.path.exists(os.path.join(output_dir, "label.pkl")):
            with open(os.path.join(output_dir, "label.pkl"), "rb") as f:
                label = pickle.load(f)
            valid_traj = label["valid"]
            use_waypoints = label["use_waypoints"]
        
        current_mask = np.load(os.path.join(output_dir, "mask.npy"))
    else:
        # User-added points start empty
        user_points = np.zeros((0, 2))
        user_labels = np.array([], dtype=int)

    DEFAULT_NO_WAYPOINTS = True
    if not DEFAULT_NO_WAYPOINTS:
        # Initialize points and labels
        # Store initial waypoints separately from user-added points
        waypoints = initial_waypoints.copy() if len(initial_waypoints) > 0 else np.zeros((0, 2))
        waypoint_labels = np.ones(len(waypoints), dtype=int)  # All initial waypoints are positive
    else:
        waypoints = np.zeros((0, 2))
        waypoint_labels = np.array([], dtype=int)
        use_waypoints = False


    # For SAM2 prediction, we need to combine both sets of points
    points = np.vstack([waypoints, user_points]) if len(waypoints) > 0 else user_points.copy()
    labels = np.concatenate([waypoint_labels, user_labels])
    
    # Keep track of user point history for undo functionality
    user_points_history = [user_points.copy()]
    user_labels_history = [user_labels.copy()]
    
    
    # Create figure for interaction
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()

    is_finished = False
    
    # Function to update the display
    def update_display():
        ax.clear()
        ax.imshow(image)

        # write valid_traj text on the top left of the image
        ax.text(10, 10, f"Valid: {valid_traj}", fontsize=12, color="red")
        ax.text(10, 20, f"Use Waypoints: {use_waypoints}", fontsize=12, color="red")
        
        # Show current mask if available
        if current_mask is not None:
            h, w = current_mask.shape
            # Make mask more transparent (0.4 instead of 0.6)
            mask_image = current_mask.reshape(h, w, 1) * np.array([30/255, 144/255, 255/255, 0.2]).reshape(1, 1, -1)
            ax.imshow(mask_image)
            
            if current_score is not None:
                ax.set_title(f"Mask Score: {current_score:.2f}", fontsize=12)
        
        # Show waypoints with red to green gradient
        if len(waypoints) > 0:
            # Sort waypoints by distance from the first point (approximating path order)
            if len(waypoints) > 1:
                # Calculate distances from first point
                distances = np.sqrt(np.sum((waypoints - waypoints[0])**2, axis=1))
                sorted_indices = np.argsort(distances)
                
                # Create color gradient from red to green
                for i, idx in enumerate(sorted_indices):
                    color_factor = i / (len(sorted_indices) - 1) if len(sorted_indices) > 1 else 0.5
                    color = (
                        (1 - color_factor),  # Red component
                        color_factor,        # Green component
                        0                    # Blue component
                    )
                    ax.scatter(waypoints[idx, 0], waypoints[idx, 1], color=color, marker='*', 
                              s=100, edgecolor='white', linewidth=1.25)
            else:
                # If only one point, make it yellow
                ax.scatter(waypoints[:, 0], waypoints[:, 1], color=(0.5, 0.5, 0), marker='*', 
                          s=100, edgecolor='white', linewidth=1.25)
        
        # Show user-added points with different markers
        if len(user_points) > 0:
            pos_points = user_points[user_labels == 1]
            neg_points = user_points[user_labels == 0]
            
            # Draw positive points in green
            if len(pos_points) > 0:
                ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', 
                          s=80, edgecolor='white', linewidth=1.25)
            
            # Draw negative points in red
            if len(neg_points) > 0:
                ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', 
                          s=80, edgecolor='white', linewidth=1.25)
        
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw()
    
    # Function to handle clicks
    def on_click(event):
        nonlocal user_points, user_labels, points, labels, current_mask, current_score, current_logits
        
        if event.inaxes != ax:
            return
            
        # Get click coordinates
        x, y = int(event.xdata), int(event.ydata)
        
        # Check if ctrl is pressed (negative point)
        is_positive = not event.key == 'control'
        
        # Save current state for undo
        user_points_history.append(user_points.copy() if len(user_points) > 0 else np.zeros((0, 2)))
        user_labels_history.append(user_labels.copy())
        
        # Add the new point to user points
        user_points = np.vstack([user_points, [x, y]]) if len(user_points) > 0 else np.array([[x, y]])
        user_labels = np.append(user_labels, 1 if is_positive else 0)
        
        # Combine waypoints and user points for prediction
        points = np.vstack([waypoints, user_points]) if len(waypoints) > 0 else user_points.copy()
        labels = np.concatenate([waypoint_labels, user_labels])
        
        # Run prediction
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                mask_input=current_logits[None, :, :] if current_logits is not None else None,
                multimask_output=True
            )
        
        # Get the best mask
        best_idx = scores.argmax()
        current_mask = masks[best_idx]
        current_score = scores[best_idx]
        current_logits = logits[best_idx]
        
        # Update display
        update_display()
    
    # Function to handle key presses
    def on_key(event):
        nonlocal current_mask, user_points, user_labels, points, labels, current_logits, \
            current_score, user_points_history, user_labels_history, is_finished, \
            waypoints, waypoint_labels, valid_traj, use_waypoints, action
        
        if event.key == ' ':  # Space to save and continue
            action = "save"
            plt.close(fig)
        elif event.key == 'z':
            valid_traj = not valid_traj
            update_display()
        elif event.key == 'q':
            is_finished = True
            # Exit without saving
            plt.close(fig)
        elif event.key == 'd':  # DELETE this frame's output directory + input assets
            action = "delete"
            plt.close(fig)
        elif event.key == 'r':  # COPY original image to ./ramp and STAY
            dst = _copy_to_ramp(frame_path, ramp_dir)
            ax.text(
                10, 40,
                f"Copied to ramp: {os.path.basename(dst) if dst else 'failed'}",
                fontsize=12, color="yellow",
                bbox=dict(facecolor='black', alpha=0.5, pad=2)
            )
            fig.canvas.draw_idle()
        elif event.key == 'c':  # clear user points & reapply initial waypoints (old 'r')
            use_waypoints = True
            waypoints = initial_waypoints.copy() if len(initial_waypoints) > 0 else np.zeros((0, 2))
            waypoint_labels = np.ones(len(waypoints), dtype=int)

            user_points = np.zeros((0, 2))
            user_labels = np.array([], dtype=int)

            points = waypoints.copy()
            labels = waypoint_labels.copy()

            user_points_history = [np.zeros((0, 2))]
            user_labels_history = [np.array([], dtype=int)]

            if len(points) > 0:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, logits = predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        mask_input=current_logits[None, :, :] if current_logits is not None else None,
                        multimask_output=True
                    )
                best_idx = scores.argmax()
                current_mask = masks[best_idx]
                current_score = scores[best_idx]
                current_logits = logits[best_idx]
            else:
                current_mask = None
                current_score = None
                current_logits = None

            update_display()
        elif event.key == 'x':
            use_waypoints = False
            # Remove all points (both waypoints and user points) and start fresh
            waypoints = np.zeros((0, 2))
            waypoint_labels = np.array([], dtype=int)

            user_points = np.zeros((0, 2))
            user_labels = np.array([], dtype=int)

            # For SAM2 prediction, we need to combine both sets of points
            points = np.vstack([waypoints, user_points]) if len(waypoints) > 0 else user_points.copy()
            labels = np.concatenate([waypoint_labels, user_labels])
            
            current_mask = None
            current_score = None
            current_logits = None
            
            # Reset history
            user_points_history = [np.zeros((0, 2))]
            user_labels_history = [np.array([], dtype=int)]
            
            update_display()
        elif event.key == 'backspace':
            # Undo last user point if there's history
            if len(user_points_history) > 1:
                user_points_history.pop()  # Remove current state
                user_labels_history.pop()
                
                # Restore previous state
                user_points = user_points_history[-1].copy()
                user_labels = user_labels_history[-1].copy()
                
                # Combine waypoints and user points for prediction
                points = np.vstack([waypoints, user_points]) if len(waypoints) > 0 and len(user_points) > 0 else \
                        waypoints.copy() if len(waypoints) > 0 else \
                        user_points.copy() if len(user_points) > 0 else \
                        np.zeros((0, 2))
                
                labels = np.concatenate([waypoint_labels, user_labels]) if len(waypoints) > 0 and len(user_points) > 0 else \
                        waypoint_labels.copy() if len(waypoints) > 0 else \
                        user_labels.copy() if len(user_points) > 0 else \
                        np.array([], dtype=int)
                
                # Re-run prediction with updated points
                if len(points) > 0:
                    print(f"shape of current_logits: {current_logits.shape}")
                    print(current_logits)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        masks, scores, logits = predictor.predict(
                            point_coords=points,
                            point_labels=labels,
                            mask_input=current_logits[None, :, :] if current_logits is not None else None,
                            multimask_output=True
                        )
                    
                    # Get the best mask
                    best_idx = scores.argmax()
                    current_mask = masks[best_idx]
                    current_score = scores[best_idx]
                    current_logits = logits[best_idx]
                else:
                    # No points left
                    current_mask = None
                    current_score = None
                    current_logits = None
                
                update_display()
    
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial prediction if we have waypoints
    if len(points) > 0 and current_mask is None:
        print(f"Initial prediction with {len(points)} points")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Run prediction
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                mask_input=current_logits[None, :, :] if current_logits is not None else None,
                multimask_output=True
            )
            print(f"scores: {scores}")
        
        # Get the best mask
        best_idx = scores.argmax()
        current_mask = masks[best_idx]
        current_score = scores[best_idx]
        current_logits = logits[best_idx]
    
    # Initial display
    update_display()
    
    # Instructions
    print("\nInteractive Segmentation Mode")
    print("-----------------------------")
    print("- Left-click: add positive (green)")
    print("- Ctrl+Left-click: add negative (red)")
    print("- Press 'c': clear user points & reapply waypoints (old 'r')")
    print("- Press 'z': toggle invalid/valid trajectory flag")
    print("- Press 'x': remove ALL points (waypoints + user)")
    print("- Press 'backspace': undo last user point")
    print("- Press 'd': delete this frame's OUTPUT DIRECTORY and continue")
    print("- Press 'r': copy this IMAGE to ./ramp and continue")
    print("- Press 'q': quit without saving")
    print("- Press SPACE: save mask/labels and continue")
    
    plt.show()
    
    if action == "save" and output_dir is not None and current_mask is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "mask.npy"), current_mask)
        with open(os.path.join(output_dir, "label.pkl"), "wb") as f:
            pickle.dump({"valid": valid_traj, "use_waypoints": use_waypoints}, f)
        if len(user_points) > 0:
            h, w = image.shape[:2]
            normalized_points = user_points.copy().astype(float)
            normalized_points[:, 0] /= float(w)
            normalized_points[:, 1] /= float(h)
            np.save(os.path.join(output_dir, "points.npy"), normalized_points)
            np.save(os.path.join(output_dir, "point_labels.npy"), user_labels)
        if save_interactive_result:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            h, w = current_mask.shape
            mask_image = current_mask.reshape(h, w, 1) * np.array([30/255, 144/255, 255/255, 0.4]).reshape(1, 1, -1)
            plt.imshow(mask_image)
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "interactive_result.png"), bbox_inches='tight')
            plt.close()

        # NEW: write dataset-style outputs under the *base* output_dir
        dataset_root = os.path.abspath(os.path.join(output_dir, os.pardir))  # parent of frame folder
        save_to_set_dirs(image, current_mask, frame_path, dataset_root=dataset_root, set_name="set1", ann_name="set1_annotation")

        _safe_rmtree(output_dir)


    elif action == "delete":
        _delete_output_dir(output_dir)
        _delete_input_assets(frame_path)
    elif action == "ramp":
        _copy_to_ramp(frame_path, ramp_dir)

    # Save the final mask if requested
    # if output_dir is not None and current_mask is not None:
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Save binary mask
    #     np.save(os.path.join(output_dir, "mask.npy"), current_mask)

    #     # save non-valid trajectory
    #     with open(os.path.join(output_dir, "label.pkl"), "wb") as f:
    #         pickle.dump({
    #             "valid": valid_traj,
    #             "use_waypoints": use_waypoints,
    #         }, f)
        
    #     # Only save user-added points (not the waypoints)
    #     if len(user_points) > 0:
    #         h, w = image.shape[:2]
    #         normalized_points = user_points.copy().astype(float)
    #         normalized_points[:, 0] /= float(w)
    #         normalized_points[:, 1] /= float(h)
            
    #         # Save normalized user points and labels
    #         np.save(os.path.join(output_dir, "points.npy"), normalized_points)
    #         np.save(os.path.join(output_dir, "point_labels.npy"), user_labels)


        
    #     # Save interactive result visualization only if requested
    #     if save_interactive_result:
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(image)
            
    #         h, w = current_mask.shape
    #         # Make mask more transparent in saved visualization too
    #         mask_image = current_mask.reshape(h, w, 1) * np.array([30/255, 144/255, 255/255, 0.4]).reshape(1, 1, -1)
    #         plt.imshow(mask_image)
            
    #         # Show waypoints with red to green gradient
    #         if len(waypoints) > 0:
    #             # Sort waypoints by distance from the first point (approximating path order)
    #             if len(waypoints) > 1:
    #                 # Calculate distances from first point
    #                 distances = np.sqrt(np.sum((waypoints - waypoints[0])**2, axis=1))
    #                 sorted_indices = np.argsort(distances)
                    
    #                 # Create color gradient from red to green
    #                 for i, idx in enumerate(sorted_indices):
    #                     color_factor = i / (len(sorted_indices) - 1) if len(sorted_indices) > 1 else 0.5
    #                     color = (
    #                         (1 - color_factor),  # Red component
    #                         color_factor,        # Green component
    #                         0                    # Blue component
    #                     )
    #                     plt.scatter(waypoints[idx, 0], waypoints[idx, 1], color=color, marker='*', 
    #                                s=100, edgecolor='white', linewidth=1.25)
    #             else:
    #                 # If only one point, make it yellow
    #                 plt.scatter(waypoints[:, 0], waypoints[:, 1], color=(0.5, 0.5, 0), marker='*', 
    #                            s=100, edgecolor='white', linewidth=1.25)
            
    #         # Show user-added points with different markers
    #         if len(user_points) > 0:
    #             pos_points = user_points[user_labels == 1]
    #             neg_points = user_points[user_labels == 0]
                
    #             # Draw positive points in green
    #             if len(pos_points) > 0:
    #                 plt.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', 
    #                            s=80, edgecolor='white', linewidth=1.25)
                
    #             # Draw negative points in red
    #             if len(neg_points) > 0:
    #                 plt.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', 
    #                            s=80, edgecolor='white', linewidth=1.25)
            
    #         plt.axis('off')
    #         plt.savefig(os.path.join(output_dir, "interactive_result.png"), bbox_inches='tight')
    #         plt.close()
    
    return current_mask, user_points, user_labels, is_finished

def process_sampled_frames(frames_dir, output_dir, limit=None, model_checkpoint="facebook/sam2-hiera-base-plus", overwrite=False):
    """
    Process all sampled frames in a directory.
    
    Args:
        frames_dir (str): Directory containing sampled frames
        output_dir (str): Directory to save output segmentations
        limit (int, optional): Limit the number of frames to process
        model_checkpoint (str): SAM2 model checkpoint to use
        overwrite (bool): Whether to overwrite existing segmentations
        
    Returns:
        dict: Statistics about the processing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all jpg files in the directory
    frame_paths = list_images(frames_dir)
    
    if limit:
        frame_paths = frame_paths[:limit]
    
    logger.info(f"Found {len(frame_paths)} frames to process")
    
    stats = {
        "total_frames": len(frame_paths),
        "processed_frames": 0,
        "failed_frames": 0,
        "skipped_frames": 0,
        "avg_waypoints_per_frame": 0,
        "avg_projected_waypoints_per_frame": 0,
        "avg_masks_per_frame": 0
    }
    
    total_waypoints = 0
    total_projected_waypoints = 0
    total_masks = 0
    
    # Initialize SAM2 predictor
    predictor = SAM2ImagePredictor.from_pretrained(model_checkpoint)
    
    # Process each frame
    for frame_path in tqdm(frame_paths):
        try:
            # Extract frame name for output
            frame_name = os.path.basename(frame_path).split('.')[0]
            frame_output_dir = os.path.join(output_dir, frame_name)
            
            # Check if this frame has already been processed
            if _dataset_ann_exists(frame_path, output_dir, ann_name="set1_annotation"):
                logger.info(f"Skipping {frame_name}: dataset mask already exists under {output_dir}/set1_annotation")
                stats["skipped_frames"] += 1
                continue
            
            os.makedirs(frame_output_dir, exist_ok=True)
            
            # Load frame and waypoints
            image, waypoints = load_sampled_frame_and_waypoints(frame_path)
            
            # Project waypoints to image
            projected_waypoints = project_waypoints_to_image(waypoints, image.shape)
            
            # Skip if no waypoints could be projected
            if len(projected_waypoints) == 0:
                logger.warning(f"No waypoints could be projected for {frame_path}")
                stats["failed_frames"] += 1
                continue
            
            # Segment with SAM2
            masks, scores, _ = segment_with_sam2(
                image,
                projected_waypoints,
                output_dir=frame_output_dir,
                visualize=True,
                predictor=predictor
            )

            # Pick best mask and save single mask.npy
            best_idx = scores.argmax()
            best_mask = masks[best_idx]
            np.save(os.path.join(frame_output_dir, "mask.npy"), best_mask)

            # Save dataset-style outputs under the base output_dir
            save_to_set_dirs(image, best_mask, frame_path, dataset_root=output_dir, set_name="set1", ann_name="set1_annotation")

            
            # Update statistics
            total_waypoints += len(waypoints)
            total_projected_waypoints += len(projected_waypoints)
            total_masks += len(masks)
            
            stats["processed_frames"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {frame_path}: {e}")
            stats["failed_frames"] += 1
    
    # Calculate averages
    if stats["processed_frames"] > 0:
        stats["avg_waypoints_per_frame"] = total_waypoints / stats["processed_frames"]
        stats["avg_projected_waypoints_per_frame"] = total_projected_waypoints / stats["processed_frames"]
        stats["avg_masks_per_frame"] = total_masks / stats["processed_frames"]
    
    # Save statistics
    with open(os.path.join(output_dir, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    
    logger.info(f"Processing complete. {stats['processed_frames']} frames processed, {stats['failed_frames']} failed, {stats['skipped_frames']} skipped.")
    logger.info(f"Average waypoints per frame: {stats['avg_waypoints_per_frame']:.2f}")
    logger.info(f"Average projected waypoints per frame: {stats['avg_projected_waypoints_per_frame']:.2f}")
    logger.info(f"Average masks per frame: {stats['avg_masks_per_frame']:.2f}")

    _safe_rmtree(frame_output_dir)
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process sampled frames with SAM2 segmentation")
    parser.add_argument("--frames_dir", default="/mnt/ihih/cityscapes/classified_output/zebra_crossing", help="Directory containing sampled frames")
    parser.add_argument("--output_dir", default="out/segmentation", help="Directory to save output segmentations")
    parser.add_argument("--limit", type=int, help="Limit the number of frames to process")
    parser.add_argument("--single_frame", help="Process a single frame instead of a directory")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode for refining segmentations")
    parser.add_argument("--compute_background", action="store_true", help="Use a second SAM2 instance to precompute next frame")
    parser.add_argument("--save_interactive_result", action="store_true", help="Save visualization of interactive segmentation results")
    parser.add_argument("--model_size", default="base", choices=["base", "small", "tiny"], 
                        help="Size of the SAM2 model to use (base, small, or tiny)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing segmentations")
    
    args = parser.parse_args()
    
    if args.interactive:
        if args.single_frame:
            # Process a single frame interactively
            image, waypoints = load_sampled_frame_and_waypoints(args.single_frame)
            projected_waypoints = project_waypoints_to_image(waypoints, image.shape)
            
            # Create output directory based on frame name
            frame_name = os.path.basename(args.single_frame).split('.')[0]
            frame_output_dir = os.path.join(args.output_dir, frame_name)
            
            # Determine model checkpoint based on size
            if args.model_size == "small":
                model_checkpoint = "facebook/sam2-hiera-small"
            elif args.model_size == "tiny":
                model_checkpoint = "facebook/sam2-hiera-tiny"
            else:  # default to base
                model_checkpoint = "facebook/sam2-hiera-base-plus"
            
            # Initialize predictor with selected model
            predictor = SAM2ImagePredictor.from_pretrained(model_checkpoint)
            predictor.set_image(image)
            
            # Run interactive segmentation
            interactive_segmentation_with_predictor(
                image,
                projected_waypoints,
                output_dir=frame_output_dir,
                predictor=predictor,
                save_interactive_result=args.save_interactive_result,
                frame_path=args.single_frame,  # <-- add this
                ramp_dir="./ramp"              # <-- optional
            )
            
            logger.info(f"Processed single frame interactively: {args.single_frame}")
        else:
            # Process all frames in directory interactively
            process_interactive_mode(
                args.frames_dir, 
                args.output_dir, 
                limit=args.limit, 
                compute_background=args.compute_background,
                save_interactive_result=args.save_interactive_result,
                model_size=args.model_size,
                overwrite=args.overwrite
            )
    elif args.single_frame:
        # Process a single frame
        image, waypoints = load_sampled_frame_and_waypoints(args.single_frame)
        projected_waypoints = project_waypoints_to_image(waypoints, image.shape)
        
        # Create output directory based on frame name
        frame_name = os.path.basename(args.single_frame).split('.')[0]
        frame_output_dir = os.path.join(args.output_dir, frame_name)
        os.makedirs(frame_output_dir, exist_ok=True)

        if _dataset_ann_exists(args.single_frame, args.output_dir, ann_name="set1_annotation"):
            logger.info(f"Skipping {os.path.basename(args.single_frame)}: dataset mask already exists under {args.output_dir}/set1_annotation")
            raise SystemExit(0)
        
        # Determine model checkpoint based on size
        if args.model_size == "small":
            model_checkpoint = "facebook/sam2-hiera-small"
        elif args.model_size == "tiny":
            model_checkpoint = "facebook/sam2-hiera-tiny"
        else:  # default to base
            model_checkpoint = "facebook/sam2-hiera-base-plus"
        
        # Segment with SAM2
        predictor = SAM2ImagePredictor.from_pretrained(model_checkpoint)
        masks, scores, _ = segment_with_sam2(
            image,
            projected_waypoints,
            output_dir=frame_output_dir,
            visualize=True,
            predictor=predictor
        )
        best_idx = scores.argmax()
        best_mask = masks[best_idx]
        # np.save(os.path.join(frame_output_dir, "mask.npy"), best_mask)
        save_to_set_dirs(image, best_mask, args.single_frame, dataset_root=args.output_dir, set_name="set1", ann_name="set1_annotation")
        
        logger.info(f"Processed single frame: {args.single_frame}")
    else:
        # Process all frames in directory
        # Determine model checkpoint based on size
        if args.model_size == "small":
            model_checkpoint = "facebook/sam2-hiera-small"
        elif args.model_size == "tiny":
            model_checkpoint = "facebook/sam2-hiera-tiny"
        else:  # default to base
            model_checkpoint = "facebook/sam2-hiera-base-plus"
            
        logger.info(f"Using SAM2 model: {model_checkpoint}")
        process_sampled_frames(args.frames_dir, args.output_dir, limit=args.limit, model_checkpoint=model_checkpoint, overwrite=args.overwrite)
