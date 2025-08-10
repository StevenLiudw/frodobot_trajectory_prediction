import time
import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor
import pickle
import glob
from tqdm import tqdm
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_output_dirs(output_root, frame_path):
    """
    Returns (set1_dir, ann_dir, frame_name).
    Creates:
      <output_root>/set1/<frame_name>/
      <output_root>/set1_annotation/<frame_name>/
    """
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    set1_dir = os.path.join(output_root, "set1", frame_name)
    ann_dir  = os.path.join(output_root, "set1_annotation", frame_name)
    os.makedirs(set1_dir, exist_ok=True)
    os.makedirs(ann_dir,  exist_ok=True)
    return set1_dir, ann_dir, frame_name

def save_mask_png(output_dir, mask, filename="mask.png"):
    os.makedirs(output_dir, exist_ok=True)
    h, w = mask.shape
    # OpenCV writes BGR; put red in channel 2
    out_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    out_bgr[..., 2] = (mask.astype(np.uint8) * 255)  # red on black
    cv2.imwrite(os.path.join(output_dir, filename), out_bgr)

def compute_output_dirs(output_root, frame_path):
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    set1_dir = os.path.join(output_root, "set1", frame_name)
    ann_dir  = os.path.join(output_root, "set1_annotation", frame_name)
    return set1_dir, ann_dir, frame_name

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
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if limit:
        frame_paths = frame_paths[:limit]

    logger.info(f"Found {len(frame_paths)} frames to process in interactive mode")

    if model_size == "small":
        model_checkpoint = "facebook/sam2-hiera-small"
    elif model_size == "tiny":
        model_checkpoint = "facebook/sam2-hiera-tiny"
    else:
        model_checkpoint = "facebook/sam2-hiera-base-plus"

    logger.info(f"Using SAM2 model: {model_checkpoint}")

    predictor1 = SAM2ImagePredictor.from_pretrained(model_checkpoint)
    predictor2 = SAM2ImagePredictor.from_pretrained(model_checkpoint) if compute_background else None

    next_valid_frame_index = None
    next_valid_frame_data = None
    precompute_thread = None

    is_first = True
    i = 0
    while i < len(frame_paths):
        try:
            frame_path = frame_paths[i]
            frame_name = os.path.basename(frame_path).split('.')[0]

            # Build paths WITHOUT creating dirs yet
            set1_dir, ann_dir, _ = compute_output_dirs(output_dir, frame_path)
            mask_png = os.path.join(ann_dir, "1.png")

            # Skip if mask already exists
            if not overwrite and os.path.exists(mask_png):
                logger.info(f"Skipping {frame_path} because mask already exists at {mask_png}")
                i += 1
                continue

            # Create ONLY the annotation dir now (we'll create set1_dir after mask is saved)
            os.makedirs(ann_dir, exist_ok=True)

            # Load frame + waypoints
            image, waypoints = load_sampled_frame_and_waypoints(frame_path)
            projected_waypoints = project_waypoints_to_image(waypoints, image.shape) if waypoints is not None else np.zeros((0, 2))

            logger.info(f"Processing {frame_path} in interactive mode")
            print(f"\nFrame: {frame_name}")

            # Predictor setup
            if is_first or not compute_background or next_valid_frame_data is None:
                predictor1.set_image(image)
                is_first = False
            else:
                if next_valid_frame_index == i:
                    predictor1, predictor2 = predictor2, predictor1
                    image, projected_waypoints = next_valid_frame_data
                    logger.info(f"Using precomputed embeddings for frame: {frame_path}")
                else:
                    predictor1.set_image(image)

            # Precompute next (unchanged except that we no longer create future dirs)
            if compute_background and predictor2 and precompute_thread is None:
                import threading
                def find_and_precompute_next_frame():
                    nonlocal next_valid_frame_index, next_valid_frame_data
                    time.sleep(0.3)
                    try:
                        next_idx = i + 1
                        while next_idx < len(frame_paths):
                            next_frame_path = frame_paths[next_idx]
                            _, next_ann_dir, _ = compute_output_dirs(output_dir, next_frame_path)
                            next_mask_png = os.path.join(next_ann_dir, "1.png")
                            if not overwrite and os.path.exists(next_mask_png):
                                next_idx += 1
                                continue
                            try:
                                next_image, next_waypoints = load_sampled_frame_and_waypoints(next_frame_path)
                                if next_waypoints is not None:
                                    next_projected_waypoints = project_waypoints_to_image(next_waypoints, next_image.shape)
                                else:
                                    next_projected_waypoints = np.zeros((0, 2))
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
                precompute_thread = threading.Thread(target=find_and_precompute_next_frame, daemon=True)
                precompute_thread.start()

            # === Interactive segmentation writes 1.png into ann_dir ===
            mask, points, labels, is_finished = interactive_segmentation_with_predictor(
                image,
                projected_waypoints,
                output_dir=ann_dir,
                predictor=predictor1,
                save_interactive_result=False
            )

            logger.info(f"Completed interactive segmentation for {frame_path}")
            if is_finished:
                logger.info("User quit the interactive segmentation.")
                return

            # If mask now exists, THEN create set1 dir and copy original as 1.jpg
            if os.path.exists(mask_png):
                os.makedirs(set1_dir, exist_ok=True)
                dst_img = os.path.join(set1_dir, "1.jpg")
                if not os.path.exists(dst_img) or overwrite:
                    shutil.copy2(frame_path, dst_img)
                    logger.info(f"Saved original image to {dst_img}")
            else:
                logger.warning(f"Mask was not saved for {frame_path}; original image will not be copied.")

            if precompute_thread and precompute_thread.is_alive():
                precompute_thread.join(timeout=0.1)
            precompute_thread = None

            i = next_valid_frame_index if next_valid_frame_index is not None else i + 1

        except Exception as e:
            logger.error(f"Error processing {frame_path} in interactive mode: {e}")
            i += 1


def interactive_segmentation_with_predictor(image, initial_waypoints, output_dir=None, predictor=None, save_interactive_result=False):
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
    assert predictor is not None, "Predictor must be provided"
    
    # Current mask state
    current_mask = None
    current_score = None
    current_logits = None

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
            waypoints, waypoint_labels, valid_traj, use_waypoints
        
        if event.key == ' ':  # Space to save and continue
            # Save the current mask and exit
            plt.close(fig)
        elif event.key == 'z':
            valid_traj = not valid_traj
            update_display()
        elif event.key == 'q':
            is_finished = True
            # Exit without saving
            plt.close(fig)
        elif event.key == 'r':
            # Reset user points but keep waypoints
            use_waypoints = True

            waypoints = initial_waypoints.copy() if len(initial_waypoints) > 0 else np.zeros((0, 2))
            waypoint_labels = np.ones(len(waypoints), dtype=int)  # All initial waypoints are positive

            user_points = np.zeros((0, 2))
            user_labels = np.array([], dtype=int)
            
            # Combine waypoints and user points for prediction
            points = waypoints.copy()
            labels = waypoint_labels.copy()
            
            # Reset user history
            user_points_history = [np.zeros((0, 2))]
            user_labels_history = [np.array([], dtype=int)]
            
            # If we have waypoints, run prediction
            if len(points) > 0:
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
    print("- Left-click to add positive points (green)")
    print("- Ctrl+Left-click to add negative points (red)")
    print("- Press 'r' to reset user-added points (keeps waypoints)")
    print("- Press 'z' to toggle invalid /valid trajectory")
    print("- Press 'x' to remove all points (including waypoints)")
    print("- Press 'backspace' to undo the last user-added point")
    print("- Press 'q' to quit without saving")
    print("- Press SPACE to save and continue to the next image")
    
    plt.show()
    
    # Save the final mask as red-on-black PNG only
    if output_dir is not None and current_mask is not None:
        save_mask_png(output_dir, current_mask, "1.png")

    
    return current_mask, user_points, user_labels, is_finished

def process_sampled_frames(frames_dir, output_dir, limit=None, model_checkpoint="facebook/sam2-hiera-base-plus", overwrite=False):
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
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
        "avg_masks_per_frame": 0,
    }

    total_waypoints = 0
    total_projected_waypoints = 0
    total_masks = 0

    predictor = SAM2ImagePredictor.from_pretrained(model_checkpoint)

    for frame_path in tqdm(frame_paths):
        try:
            # Build paths WITHOUT creating dirs yet
            set1_dir, ann_dir, _ = compute_output_dirs(output_dir, frame_path)
            mask_png = os.path.join(ann_dir, "1.png")

            # Skip if mask already exists
            if not overwrite and os.path.exists(mask_png):
                logger.info(f"Skipping {frame_path} as it has already been processed")
                stats["skipped_frames"] += 1
                continue

            # Create ONLY annotation dir now
            os.makedirs(ann_dir, exist_ok=True)

            # Load frame & waypoints
            image, waypoints = load_sampled_frame_and_waypoints(frame_path)
            projected_waypoints = project_waypoints_to_image(waypoints, image.shape)

            if len(projected_waypoints) == 0:
                logger.warning(f"No waypoints could be projected for {frame_path}")
                stats["failed_frames"] += 1
                continue

            # Segment
            masks, scores, _ = segment_with_sam2(
                image,
                projected_waypoints,
                output_dir=None,
                visualize=True,
                predictor=predictor
            )

            if masks is None or len(masks) == 0:
                logger.warning(f"No mask predicted for {frame_path}")
                stats["failed_frames"] += 1
                continue

            # Save best mask to 1.png
            best_idx = scores.argmax()
            best_mask = masks[best_idx]
            save_mask_png(ann_dir, best_mask, "1.png")

            # Only AFTER mask exists, copy original to set1/<frame>/1.jpg
            if os.path.exists(mask_png):
                os.makedirs(set1_dir, exist_ok=True)
                dst_img = os.path.join(set1_dir, "1.jpg")
                if not os.path.exists(dst_img) or overwrite:
                    shutil.copy2(frame_path, dst_img)

            total_waypoints += len(waypoints)
            total_projected_waypoints += len(projected_waypoints)
            total_masks += len(masks)
            stats["processed_frames"] += 1

        except Exception as e:
            logger.error(f"Error processing {frame_path}: {e}")
            stats["failed_frames"] += 1

    if stats["processed_frames"] > 0:
        stats["avg_waypoints_per_frame"] = total_waypoints / stats["processed_frames"]
        stats["avg_projected_waypoints_per_frame"] = total_projected_waypoints / stats["processed_frames"]
        stats["avg_masks_per_frame"] = total_masks / stats["processed_frames"]

    with open(os.path.join(output_dir, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    logger.info(f"Processing complete. {stats['processed_frames']} frames processed, {stats['failed_frames']} failed, {stats['skipped_frames']} skipped.")
    logger.info(f"Average waypoints per frame: {stats['avg_waypoints_per_frame']:.2f}")
    logger.info(f"Average projected waypoints per frame: {stats['avg_projected_waypoints_per_frame']:.2f}")
    logger.info(f"Average masks per frame: {stats['avg_masks_per_frame']:.2f}")

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
                save_interactive_result=args.save_interactive_result
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
        
        # Determine model checkpoint based on size
        if args.model_size == "small":
            model_checkpoint = "facebook/sam2-hiera-small"
        elif args.model_size == "tiny":
            model_checkpoint = "facebook/sam2-hiera-tiny"
        else:  # default to base
            model_checkpoint = "facebook/sam2-hiera-base-plus"
        
        # Segment with SAM2
        predictor = SAM2ImagePredictor.from_pretrained(model_checkpoint)
        segment_with_sam2(
            image, 
            projected_waypoints, 
            output_dir=frame_output_dir,
            visualize=True,
            predictor=predictor
        )
        
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
