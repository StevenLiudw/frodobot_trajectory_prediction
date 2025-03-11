import os
import pickle as pkl
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from torchvision import transforms
import logging
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrajInvalidException(Exception):
    """Exception raised when a trajectory is invalid."""
    pass

class TrajectoryDataset(Dataset):
    """Dataset for training a model to predict future waypoints given an RGB image and goal."""
    
    def __init__(
        self,
        data_dir,
        ride_dirs=None,
        n_waypoints=10,
        transform=None,
        p1=0.2,  # Prob for strategy 1: use one of the next N waypoints as goal
        p2=0.4,  # Prob for strategy 2: sample beyond N waypoints, no U-turns
        p3=0.4,  # Prob for strategy 3: sample in general heading direction
        max_n_waypoints=200, # max number of waypoints to consider for strategy 2
        min_n_waypoints=20, # min number of waypoints to consider for strategy 2
        max_goal_distance=500, # max distance for sampled goals (strategy 3)
        min_goal_distance=10, # min distance for sampled goals (strategy 3)
        min_goal_distance_strategy2=4, # roughly the max dist of strategy 1
        goal_angle_threshold=30, # max angle deviation in degrees (strategy 3)
        alignment_threshold_deg=30, # max angle deviation in degrees (strategy 2)
        dummy_goal=False, # whether to use dummy goals for testing
        seed=42
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Path to the data directory
            n_waypoints (int): Number of future waypoints to predict
            transform (callable, optional): Transform to apply to the images
            p1, p2, p3 (float): Probabilities for each goal sampling strategy
            max_goal_distance (float): Maximum distance for sampled goals (strategy 3)
            min_goal_distance (float): Minimum distance for sampled goals (strategy 3)
            goal_angle_threshold (float): Maximum angle deviation in degrees (strategy 3)
            dummy_goal (bool): Whether to use dummy goals for testing
            seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.n_waypoints = n_waypoints
        self.transform = transform
        self._dummy_goal = dummy_goal
        
        # Validate strategy probabilities
        assert abs(p1 + p2 + p3 - 1.0) < 1e-6, "Strategy probabilities must sum to 1"
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        
        # for strategy 2
        self.max_n_waypoints = max_n_waypoints
        self.min_n_waypoints = min_n_waypoints
        self.min_goal_distance_strategy2 = min_goal_distance_strategy2
        self.alignment_threshold_deg = alignment_threshold_deg
        self.max_goal_distance = max_goal_distance
        self.min_goal_distance = min_goal_distance
        self.goal_angle_threshold = goal_angle_threshold
        
        # Set random seed for reproducibility
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if ride_dirs is None:
            # Find all ride directories
            self.ride_dirs = sorted(glob.glob(os.path.join(data_dir, "ride_*")))
            logger.info(f"Found {len(self.ride_dirs)} ride directories")
        else:
            self.ride_dirs = ride_dirs
        
        # Preprocess and index the valid samples
        if os.path.exists(f"{self.data_dir}/preprocessed/samples.pkl"):
            with open(f"{self.data_dir}/preprocessed/samples.pkl", "rb") as f:
                self.samples = pkl.load(f)
        else:
            self.samples = None
            self._preprocess_data()
        
        logger.info(f"Initialized dataset with {len(self.samples)} valid samples")
        
    def _preprocess_data(self):
        """Preprocess and index all valid data samples."""
        logger.info("Preprocessing data...")
        sample_count = 0
        samples = []
        
        for ride_dir in tqdm(self.ride_dirs):

            ride_name = os.path.basename(ride_dir)
            traj_path = os.path.join(ride_dir, "traj_data.pkl")
            
            try:
                # Skip if trajectory data doesn't exist
                if not os.path.exists(traj_path):
                    logger.warning(f"No trajectory data found for {ride_name}")
                    raise TrajInvalidException(f"No trajectory data found for {ride_name}")
                
                # Load trajectory data
                with open(traj_path, "rb") as f:
                    traj_data = pkl.load(f)
                    
                pos = traj_data["pos"]  # Nx2 array of positions
                    
                # Check the image directory
                img_dir = os.path.join(ride_dir, "img")
                if not os.path.exists(img_dir):
                    logger.warning(f"No image directory found for {ride_name}")
                    raise TrajInvalidException(f"No image directory found for {ride_name}")
                    
                # Find all valid samples in this ride
                max_idx = len(pos) - 2 * self.n_waypoints # leave room for goal sampling
                if max_idx < 0:
                    logger.warning(f"Not enough data in {ride_name} for goal sampling")
                    raise TrajInvalidException(f"Not enough data in {ride_name} for goal sampling")

                valid_indices = []
                for i in range(max_idx):
                    img_path = os.path.join(img_dir, f"{i}.jpg")
                    if os.path.exists(img_path):
                        valid_indices.append(i)
                
                logger.debug(f"Found {len(valid_indices)} valid frames in {ride_name}")
                
                # Create sample entries
                for i in valid_indices:
                    samples.append({
                        "ride_dir": ride_dir,
                        "img_idx": i,
                        "traj_data_path": traj_path
                    })
                    sample_count += 1
                
            except TrajInvalidException as e:
                logger.warning(f"Skipping {ride_name} due to invalid trajectory data: {e}")
                # remove the ride from the list
                self.ride_dirs.remove(ride_dir)
                # remove the dir
                # shutil.rmtree(ride_dir, ignore_errors=True)
                logger.info(f"Removed {ride_name} due to invalid trajectory data")
                
                continue
            except Exception as e:
                logger.error(f"Error loading trajectory data for {ride_name}: {e}")
                continue

        logger.info(f"Total valid samples: {sample_count}")

        # convert to numpy array to avoid memory growing for multiple workers
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.samples = np.array(samples) 

        # save the samples to a file
        os.makedirs(f"{self.data_dir}/preprocessed", exist_ok=True)
        with open(f"{self.data_dir}/preprocessed/samples.pkl", "wb") as f:
            pkl.dump(self.samples, f)
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
        
    def _transform_to_local_frame(self, waypoints, current_pos, current_yaw):
        """
        Transform waypoints from global to local reference frame.
        
        Args:
            waypoints (np.ndarray): Array of waypoints (Nx2)
            current_pos (np.ndarray): Current position (x, y)
            current_yaw (float): Current orientation in radians
            
        Returns:
            np.ndarray: Transformed waypoints in local frame
        """
        # Ensure inputs are numpy arrays
        waypoints = np.asarray(waypoints)
        current_pos = np.asarray(current_pos)
        
        # Translate waypoints relative to current position
        translated = waypoints - current_pos
        
        # Create rotation matrix
        # Note: we rotate counter to the vehicle orientation to get local frame
        cos_theta = np.cos(-current_yaw)
        sin_theta = np.sin(-current_yaw)
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        # Apply rotation (vectorized operation)
        transformed = np.matmul(translated, R.T)
        
        return transformed
        
    def _detect_u_turn(self, traj_data, start_idx, end_idx, threshold_angle=120):
        """
        Detect if there's a U-turn between start_idx and end_idx.
        
        Args:
            traj_data (dict): Trajectory data
            start_idx (int): Start index
            end_idx (int): End index
            threshold_angle (float): Angle threshold in degrees to consider a U-turn
            
        Returns:
            bool: True if U-turn detected, False otherwise
        """
        # Validate indices
        if end_idx <= start_idx:
            return True
            
        pos = traj_data["pos"]
        yaw = traj_data["yaw"]
        
        if start_idx >= len(pos) or end_idx >= len(pos):
            return True
            
        # Method 1: Check for large yaw changes - VECTORIZED
        # Get all relevant yaw angles
        yaw_segment = yaw[start_idx+1:end_idx+1]
        start_yaw = yaw[start_idx]
        
        # Calculate absolute angle differences
        angle_diffs = np.abs(yaw_segment - start_yaw)
        # Handle wrap-around (convert to degrees)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs) * 180.0 / np.pi
        
        # Check if any angle difference exceeds the threshold
        if np.any(angle_diffs > threshold_angle):
            return True
                
        # Method 2: Check path characteristics
        # Extract the segment of the path
        path_segment = pos[start_idx:end_idx+1]
        
        # If the path is very short, no need for additional checks
        if len(path_segment) < 3:
            return False
            
        # Calculate the total path length
        segment_diffs = np.diff(path_segment, axis=0)
        path_length = np.sum(np.sqrt(np.sum(segment_diffs**2, axis=1)))
        
        # Calculate the direct distance from start to end
        direct_distance = np.sqrt(np.sum((path_segment[-1] - path_segment[0])**2))
        
        # If the path is much longer than the direct distance, it might indicate a U-turn
        if path_length > 2.0 * direct_distance:
            return True
            
        return False
        
    def _sample_goal_strategy1(self, traj_data, current_idx):
        """
        Strategy 1: Sample a goal from within the next N waypoints.
        Adjust waypoints after the goal to match the goal position.
        
        Args:
            traj_data (dict): Trajectory data
            current_idx (int): Current index in the trajectory
            
        Returns:
            tuple: (goal_pos, waypoints) or None if invalid
        """
        pos = traj_data["pos"]
        
        # Get the next N waypoints
        next_waypoints = pos[current_idx+1:current_idx+self.n_waypoints+1].copy()
        
        # Sample a goal from one of these waypoints
        goal_idx_relative = random.randint(0, self.n_waypoints - 1)
        goal_idx_absolute = current_idx + 1 + goal_idx_relative
        goal_pos = pos[goal_idx_absolute].copy()
        
        # Adjust waypoints after the goal to match the goal position
        for i in range(goal_idx_relative, self.n_waypoints):
            next_waypoints[i] = goal_pos
            
        return goal_pos, next_waypoints
        
    def _sample_goal_strategy2(self, traj_data, current_idx):
        """
        Strategy 2: Use the furthest future waypoint that's aligned with the robot's heading direction.
        
        Args:
            traj_data (dict): Trajectory data
            current_idx (int): Current index in the trajectory
            
        Returns:
            tuple: (goal_pos, waypoints) or None if invalid
        """
        pos = traj_data["pos"]
        
        # Get the next N waypoints
        next_waypoints = pos[current_idx+1:current_idx+self.n_waypoints+1].copy()
        
        # Get current position
        current_pos = pos[current_idx].copy()
        
        # Calculate heading direction accounting for turns
        heading_angle = self._calculate_heading_direction(traj_data, current_idx)
        
        # Define range for goal search
        min_idx = current_idx + self.min_n_waypoints
        max_idx = min(current_idx + self.max_n_waypoints, len(pos) - 1)
        
        if min_idx >= max_idx:
            return None  # Not enough future waypoints
            
        
        
        # Vectorized implementation
        # Get all waypoints in the range
        waypoints_range = pos[min_idx:max_idx]
        
        # Calculate vectors to all waypoints
        directions = waypoints_range - current_pos
        
        # # Calculate distances to all waypoints
        distances = np.sqrt(np.sum(directions**2, axis=1))
        
        # Calculate angles to all waypoints
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        
        # Calculate angle differences, handling wrap-around
        angle_diffs = np.abs(angles - heading_angle)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        angle_diffs_deg = np.degrees(angle_diffs)
        
        # Find waypoints that are aligned and far enough
        valid_mask = (angle_diffs_deg <= self.alignment_threshold_deg) & \
            (distances >= self.min_goal_distance_strategy2)
        
        if np.any(valid_mask):
            # simply use the last valid waypoint
            last_idx = np.where(valid_mask)[0][-1]
            furthest_aligned_idx = np.arange(min_idx, max_idx)[last_idx]
            # max_aligned_distance = distances[last_idx]

            goal_pos = pos[furthest_aligned_idx].copy()
            return goal_pos, next_waypoints

        # Fall back to strategy 3 if no aligned waypoint found
        return self._sample_goal_strategy3(traj_data, current_idx)
        
    def _calculate_heading_direction(self, traj_data, current_idx, num_final_points=5):
        """
        Calculate the true heading direction by focusing on the final points of the trajectory.
        
        Args:
            traj_data (dict): Trajectory data
            current_idx (int): Current index in the trajectory
            num_final_points (int): Number of final points to use for direction calculation
            
        Returns:
            float: Final heading direction angle in radians
        """
        pos = traj_data["pos"]
        yaw = traj_data["yaw"]
        end_idx = current_idx + self.n_waypoints
        
        # If the range is invalid, use current yaw
        if end_idx <= current_idx:
            return yaw[current_idx]
            
        # Get all relevant positions in the range
        relevant_positions = pos[current_idx:end_idx+1]
        
        # If not enough waypoints, use current yaw
        if len(relevant_positions) < 3:
            return yaw[current_idx]
            
        # Determine how many final points to use (based on available points)
        points_to_use = min(num_final_points, len(relevant_positions) - 1)
        
        # Extract the final segment of the path
        final_segment = relevant_positions[-points_to_use:]
        
        # Calculate the direction vector from the first to last point of the final segment
        direction = final_segment[-1] - final_segment[0]
        
        # Calculate the heading angle
        heading_angle = np.arctan2(direction[1], direction[0])
        
        return heading_angle
    
    def _sample_goal_strategy3(self, traj_data, current_idx):
        """
        Strategy 3: Sample a goal in the general heading direction, accounting for turns.
        
        Args:
            traj_data (dict): Trajectory data
            current_idx (int): Current index in the trajectory
            
        Returns:
            tuple: (goal_pos, waypoints) or None if invalid
        """
        pos = traj_data["pos"]
        
        # Get the next N waypoints
        next_waypoints = pos[current_idx+1:current_idx+self.n_waypoints+1].copy()
        
        # Get current position and heading
        current_pos = pos[current_idx].copy()
        
        # Calculate heading direction accounting for turns
        direction_angle = self._calculate_heading_direction(traj_data, current_idx)
            
        # Sample distance and angle
        distance = random.uniform(self.min_goal_distance, self.max_goal_distance)
        angle_deviation = np.radians(random.uniform(-self.goal_angle_threshold, self.goal_angle_threshold))
        final_angle = direction_angle + angle_deviation
        
        # Calculate goal position
        goal_x = current_pos[0] + distance * np.cos(final_angle)
        goal_y = current_pos[1] + distance * np.sin(final_angle)
        goal_pos = np.array([goal_x, goal_y])
        
        return goal_pos, next_waypoints
        
    def _sample_goal(self, traj_data, current_idx):
        """
        Sample a goal using one of the three strategies based on probabilities.
        
        Args:
            traj_data (dict): Trajectory data
            current_idx (int): Current index in the trajectory
            
        Returns:
            tuple: (goal_pos, waypoints) or None if all strategies fail
        """
        if self._dummy_goal:
            return self._create_dummy_goal(traj_data, current_idx)
        
        # first detect u-turns
        # if there is a u-turn, use strategy 1
        if self._detect_u_turn(traj_data, current_idx, current_idx + self.n_waypoints):
            result = self._sample_goal_strategy1(traj_data, current_idx)
            return result
        
        # Randomly select a strategy based on probabilities
        strategy = random.choices([1, 2, 3], weights=[self.p1, self.p2, self.p3])[0]
        
        # Try the selected strategy first
        if strategy == 1:
            result = self._sample_goal_strategy1(traj_data, current_idx)
        elif strategy == 2:
            result = self._sample_goal_strategy2(traj_data, current_idx)
        else:  # strategy == 3
            result = self._sample_goal_strategy3(traj_data, current_idx)
            
        # If the selected strategy fails, use strategy 1
        if result is None:
            result = self._sample_goal_strategy1(traj_data, current_idx)
                    
        return result
        
    def _load_image(self, img_path):
        """
        Load and preprocess an image.
        
        Args:
            img_path (str): Path to the image
            
        Returns:
            torch.Tensor or None: Processed image or None if loading fails
        """
        try:
            # Try to load the image
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                return None
                
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transform if provided
            if self.transform:
                img = self.transform(img)
            else:
                # Default normalization
                img = torch.from_numpy(img).float() / 255.0
                img = img.permute(2, 0, 1)  # HWC to CHW
                
            return img
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return None
            
    def _create_invalid_sample(self):
        """Create an invalid sample placeholder."""
        return {
            'image': torch.zeros((3, 224, 224)),
            'waypoints': torch.zeros((self.n_waypoints, 2)),
            'goal': torch.zeros(2),
            'is_valid': False
        }
            
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample data including image, waypoints, and goal
        """
            
        try:
            # Get sample info
            sample_info = self.samples[idx]
            ride_dir = sample_info["ride_dir"]
            img_idx = sample_info["img_idx"]
            traj_data_path = sample_info["traj_data_path"]
            
            # Load trajectory data
            with open(traj_data_path, "rb") as f:
                traj_data = pkl.load(f)
                
            # Get current position and orientation
            current_pos = traj_data["pos"][img_idx]
            current_yaw = traj_data["yaw"][img_idx]
            
            # Load image
            img_path = os.path.join(ride_dir, "img", f"{img_idx}.jpg")
            image = self._load_image(img_path)
            
            if image is None:
                # Return an invalid sample
                return self._create_invalid_sample()
                
            # Check for preprocessed waypoints
            preprocessed_path = os.path.join(ride_dir, "preprocessed", f"waypoints_{img_idx}.npy")
            
            # Sample goal and get waypoints
            goal_result = self._sample_goal(traj_data, img_idx)
            
            if goal_result is None:
                return self._create_invalid_sample()
                
            goal_pos, next_waypoints = goal_result
            
            # Transform waypoints to local frame
            local_waypoints = self._transform_to_local_frame(next_waypoints, current_pos, current_yaw)
                
            if goal_result is None:
                return self._create_invalid_sample()
                
            goal_pos, _ = goal_result
            
            # Transform goal to local frame
            local_goal = self._transform_to_local_frame(goal_pos.reshape(1, 2), current_pos, current_yaw)[0]
            
            # Convert to tensors
            local_waypoints_tensor = torch.from_numpy(local_waypoints).float()
            local_goal_tensor = torch.from_numpy(local_goal).float()
            
            # Create sample dictionary
            sample = {
                'image': image,
                'waypoints': local_waypoints_tensor,
                'goal': local_goal_tensor,
                'is_valid': True
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            return self._create_invalid_sample()
            
    def collate_fn(self, batch):
        """
        Custom collate function to handle invalid samples.
        
        Args:
            batch (list): List of samples
            
        Returns:
            dict: Batched samples
        """
        # Filter out invalid samples
        valid_samples = [sample for sample in batch if sample['is_valid']]
        
        if not valid_samples:
            # # Return a dummy batch if all samples are invalid
            # return {
            #     'image': torch.zeros((1, 3, 224, 224)),
            #     'waypoints': torch.zeros((1, self.n_waypoints, 2)),
            #     'goal': torch.zeros((1, 2)),
            #     'is_valid': torch.zeros(1, dtype=torch.bool)
            # }
            return None
            
        # Stack valid samples
        images = torch.stack([sample['image'] for sample in valid_samples])
        waypoints = torch.stack([sample['waypoints'] for sample in valid_samples])
        goals = torch.stack([sample['goal'] for sample in valid_samples])
        
        return {
            'image': images,
            'waypoints': waypoints,
            'goal': goals,
        }

    def _create_dummy_goal(self, traj_data, current_idx):
        """
        Create a dummy goal for testing purposes.
        
        Args:
            traj_data (dict): Trajectory data
            current_idx (int): Current index in the trajectory
            
        Returns:
            tuple: (goal_pos, waypoints)
        """
        pos = traj_data["pos"]
        
        # Get the next N waypoints
        next_waypoints = pos[current_idx+1:current_idx+self.n_waypoints+1].copy()
        
        goal_pos = np.array([10, 10])
        
        return goal_pos, next_waypoints
    
    def shuffle(self):
        random.shuffle(self.samples)


def create_trajectory_dataloader(
    data_dir,
    batch_size=32,
    n_waypoints=10,
    image_size=(224, 224),
    num_workers=4,
    shuffle=True,
    p1=0.3,
    p2=0.3,
    p3=0.4,
    seed=42
):
    """
    Create a DataLoader for the trajectory dataset.
    
    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size
        n_waypoints (int): Number of future waypoints to predict
        image_size (tuple): Size to resize images to
        num_workers (int): Number of worker threads for data loading
        shuffle (bool): Whether to shuffle the data
        p1, p2, p3 (float): Probabilities for each goal sampling strategy
        preprocess (bool): Whether to preprocess data for faster loading
        cache_size (int): Number of samples to cache in memory
        seed (int): Random seed for reproducibility
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the trajectory dataset
    """
    # Define image transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        n_waypoints=n_waypoints,
        transform=transform,
        p1=p1,
        p2=p2,
        p3=p3,
        seed=seed
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        pin_memory=True
    )
    
    return dataloader


def visualize_trajectory_samples(
    data_dir,
    output_dir="./visualization_output",
    num_samples=5,
    n_waypoints=20,
    seed=42,
    camera_height=0.561,  # meters above ground
    camera_pitch=0
):
    """
    Visualize samples from the trajectory dataset.
    
    Args:
        data_dir (str): Path to the data directory
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
        n_waypoints (int): Number of waypoints
        seed (int): Random seed
        camera_height (float): Height of camera above ground
        camera_pitch (float): Camera pitch in radians
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define image transforms for loading the original images
    # For visualization, we need both normalized and unnormalized versions
    unnormalized_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    # Set up dataset and dataloader
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        n_waypoints=n_waypoints,
        transform=unnormalized_transform,
        seed=seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time for visualization
        shuffle=True,
        num_workers=0,  # Use single process for visualization
        collate_fn=dataset.collate_fn
    )
    
    # Camera intrinsic matrix (assumed from the reference code)
    # K = np.array([
    #     [407.86, 0, 533.301], 
    #     [0, 407.866, 278.699], 
    #     [0, 0, 1]
    # ])
    # K[0, :] /= 2 # downsample the image
    # K[1, :] /= 2 # downsample the image
    K = np.array([
        [203.93, 0, 192], 
        [0, 203.933, 144], 
        [0, 0, 1]
    ])
    
    # Initialize sample counter
    sample_count = 0
    
    # Iterate through batches
    for batch_idx, batch in enumerate(dataloader):
        # Skip if batch is None (all invalid samples)
        if batch is None:
            continue
            
        # Extract batch data (batch size is 1)
        image = batch['image'][0]  # Shape: [3, H, W]
        waypoints = batch['waypoints'][0]  # Shape: [N, 2]
        goal = batch['goal'][0]  # Shape: [2]
        
        # Convert image tensor to numpy for visualization
        original_image = image.permute(1, 2, 0).numpy()  # [H, W, 3]
        original_image = (original_image * 255).astype(np.uint8)  # Scale back to 0-255
        
        # Create a copy for drawing
        image_with_waypoints = original_image.copy()
        
        # Current position is assumed to be at origin in local frame
        current_pos_local = np.array([0, 0])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Left subplot: Image with projected waypoints
        # Project waypoints onto image
        for i, wp in enumerate(waypoints):
                
            wp_camera = np.array([-wp[1], camera_height, wp[0]])
            
            # Skip points behind the camera
            if wp_camera[2] <= 0:
                continue
                
            # Project to image plane
            wp_image = K @ wp_camera
            wp_image = wp_image / wp_camera[2]
            
            x, y = int(wp_image[0]), int(wp_image[1])
            
            # Check if point is within image bounds
            if 0 <= x < image_with_waypoints.shape[1] and 0 <= y < image_with_waypoints.shape[0]:
                # Draw waypoint with color based on distance (red -> yellow -> green)
                # Closer points are red, farther points are green
                color_factor = i / len(waypoints)
                color = (
                    int(255 * (1 - color_factor)),  # R
                    int(255 * color_factor),        # G
                    0                               # B
                )
                cv2.circle(image_with_waypoints, (x, y), 5, color, -1)
        
        # Display image with projected waypoints
        ax1.imshow(image_with_waypoints)
        ax1.set_title("Image with Projected Waypoints")
        ax1.axis('off')
        
        # Right subplot: Bird's-eye view
        # Plot current position (blue)
        ax2.scatter(current_pos_local[0], current_pos_local[1], color='blue', marker='o', s=100, label='Current Position')
        
        # Plot waypoints (colorful gradient)
        for i, wp in enumerate(waypoints):
            if isinstance(wp, torch.Tensor):
                wp = wp.numpy()
            color_factor = i / len(waypoints)
            ax2.scatter(
                wp[0], wp[1], 
                color=(1-color_factor, color_factor, 0),  # RGB
                marker='x', s=50
            )
            # Connect waypoints with a line
            if i > 0:
                prev_wp = waypoints[i-1].numpy() if isinstance(waypoints[i-1], torch.Tensor) else waypoints[i-1]
                ax2.plot([prev_wp[0], wp[0]], [prev_wp[1], wp[1]], 'k--', alpha=0.3)
        
        # Plot goal (red star)
        goal_np = goal.numpy() if isinstance(goal, torch.Tensor) else goal
        ax2.scatter(goal_np[0], goal_np[1], color='red', marker='*', s=200, label='Goal')
        
        # Add arrowhead to show orientation
        # ax2.arrow(0, 0, 1, 0, head_width=1, head_length=1, fc='blue', ec='blue')
        
        # Set equal aspect ratio and add grid
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.set_title("Waypoints and Goal (Local Frame)")
        ax2.set_xlabel("x (forward)")
        ax2.set_ylabel("y (left)")
        ax2.legend()
        
        # Set reasonable limits for the plot
        max_range = max(
            abs(waypoints[:, 0].max()), abs(waypoints[:, 0].min()),
            abs(waypoints[:, 1].max()), abs(waypoints[:, 1].min()),
            abs(goal_np[0]), abs(goal_np[1])
        ) * 1.2  # Add some margin
        
        ax2.set_xlim(-max_range/4, max_range)
        ax2.set_ylim(-max_range/2, max_range/2)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{sample_count}.png"), dpi=150)
        plt.close(fig)
        
        sample_count += 1
        print(f"Processed sample {sample_count}/{num_samples}")
        
        # Stop after processing the requested number of samples
        if sample_count >= num_samples:
            break
    
    print(f"Visualization complete. {sample_count} images saved to {output_dir}")


def analyze_waypoint_statistics(
    data_dir,
    output_dir="out/analysis",
    n_waypoints=10,
    num_samples=10000,
    batch_size=200,
    seed=42
):
    """
    Analyze statistics of waypoint trajectories using PyTorch with CUDA acceleration.
    
    Args:
        data_dir (str): Path to the data directory
        output_dir (str): Directory to save analysis results
        n_waypoints (int): Number of waypoints to analyze
        num_samples (int): Number of samples to process
        batch_size (int): Batch size for processing
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Statistics including mean and std of path lengths
    """
    # Set up dataset
    dataset = TrajectoryDataset(
        data_dir=data_dir,
        n_waypoints=n_waypoints,
        transform=None,
        seed=seed,
        dummy_goal=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        pin_memory=True  # Speed up data transfer to GPU
    )
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize tensors to store measurements
    total_path_lengths = []
    direct_distances = []
    segment_distances_all = []  # Store all segment distances for all samples
    
    # Process samples
    sample_count = 0
    logger.info(f"Analyzing waypoint statistics for {n_waypoints} waypoints...")
    
    for batch in tqdm(dataloader, total=min(num_samples // batch_size + 1, len(dataloader))):
        # Skip if batch is None (all invalid samples)
        if batch is None:
            continue
            
        # Extract waypoints and move to device
        waypoints_batch = batch['waypoints'].to(device)  # Shape: [B, N, 2]
        batch_size_actual = waypoints_batch.shape[0]
        
        # Calculate segments between consecutive waypoints
        segments = waypoints_batch[:, 1:] - waypoints_batch[:, :-1]  # [B, N-1, 2]
        segment_lengths = torch.sqrt(torch.sum(segments**2, dim=2))  # [B, N-1]
        
        # Calculate total path lengths
        batch_path_lengths = torch.sum(segment_lengths, dim=1)  # [B]
        
        # Calculate direct distances (start to end)
        batch_direct_distances = torch.sqrt(torch.sum(
            (waypoints_batch[:, -1] - waypoints_batch[:, 0])**2, 
            dim=1
        ))  # [B]
        
        # Store all segment distances for statistics
        segment_distances_all.append(segment_lengths.cpu())
        
        # Move results back to CPU and convert to numpy for storage
        total_path_lengths.extend(batch_path_lengths.cpu().numpy().tolist())
        direct_distances.extend(batch_direct_distances.cpu().numpy().tolist())
        
        sample_count += batch_size_actual
        if sample_count >= num_samples:
            # Trim to exact number of samples requested
            total_path_lengths = total_path_lengths[:num_samples]
            direct_distances = direct_distances[:num_samples]
            sample_count = num_samples
            break
    
    # Convert to numpy arrays for statistics calculation
    total_path_lengths = np.array(total_path_lengths)
    direct_distances = np.array(direct_distances)
    
    # Concatenate all segment distances and convert to numpy
    all_segments = torch.cat(segment_distances_all, dim=0).numpy()
    
    # Calculate per-segment statistics
    segment_stats = {
        'mean': float(np.mean(all_segments)),
        'std': float(np.std(all_segments)),
        'min': float(np.min(all_segments)),
        'max': float(np.max(all_segments)),
        'median': float(np.median(all_segments)),
        'per_position': []
    }
    
    # Calculate statistics for each segment position
    for i in range(all_segments.shape[1]):
        segment_i = all_segments[:, i]
        segment_stats['per_position'].append({
            'position': i,
            'mean': float(np.mean(segment_i)),
            'std': float(np.std(segment_i)),
            'min': float(np.min(segment_i)),
            'max': float(np.max(segment_i)),
            'median': float(np.median(segment_i))
        })
    
    # Calculate path directness ratio (direct distance / total path length)
    directness_ratios = direct_distances / total_path_lengths
    
    stats = {
        'total_path_length': {
            'mean': float(np.mean(total_path_lengths)),
            'std': float(np.std(total_path_lengths)),
            'min': float(np.min(total_path_lengths)),
            'max': float(np.max(total_path_lengths)),
            'median': float(np.median(total_path_lengths))
        },
        'direct_distance': {
            'mean': float(np.mean(direct_distances)),
            'std': float(np.std(direct_distances)),
            'min': float(np.min(direct_distances)),
            'max': float(np.max(direct_distances)),
            'median': float(np.median(direct_distances))
        },
        'directness_ratio': {
            'mean': float(np.mean(directness_ratios)),
            'std': float(np.std(directness_ratios)),
            'min': float(np.min(directness_ratios)),
            'max': float(np.max(directness_ratios)),
            'median': float(np.median(directness_ratios))
        },
        'segment_distances': segment_stats,
        'num_samples': sample_count
    }
    
    # Print summary
    logger.info(f"Waypoint Statistics Summary (n={sample_count}):")
    logger.info(f"Total Path Length: {stats['total_path_length']['mean']:.2f} ± {stats['total_path_length']['std']:.2f}")
    logger.info(f"Direct Distance: {stats['direct_distance']['mean']:.2f} ± {stats['direct_distance']['std']:.2f}")
    logger.info(f"Directness Ratio: {stats['directness_ratio']['mean']:.2f} ± {stats['directness_ratio']['std']:.2f}")
    logger.info(f"Segment Distance: {stats['segment_distances']['mean']:.2f} ± {stats['segment_distances']['std']:.2f}")
    
    # Create histogram plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(total_path_lengths, bins=30, alpha=0.7)
    axes[0, 0].set_title('Total Path Length Distribution')
    axes[0, 0].set_xlabel('Length')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(direct_distances, bins=30, alpha=0.7)
    axes[0, 1].set_title('Direct Distance Distribution')
    axes[0, 1].set_xlabel('Distance')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[1, 0].hist(directness_ratios, bins=30, alpha=0.7)
    axes[1, 0].set_title('Directness Ratio Distribution')
    axes[1, 0].set_xlabel('Ratio (Direct/Total)')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(all_segments.flatten(), bins=30, alpha=0.7)
    axes[1, 1].set_title('Segment Distance Distribution')
    axes[1, 1].set_xlabel('Distance')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"waypoint_stats_n{n_waypoints}.png"), dpi=150)
    plt.close(fig)
    
    # Create a plot for segment distances by position
    plt.figure(figsize=(12, 6))
    positions = [stat['position'] for stat in segment_stats['per_position']]
    means = [stat['mean'] for stat in segment_stats['per_position']]
    stds = [stat['std'] for stat in segment_stats['per_position']]
    
    plt.errorbar(positions, means, yerr=stds, fmt='o-', capsize=5)
    plt.title('Segment Distance by Position')
    plt.xlabel('Segment Position')
    plt.ylabel('Distance (mean ± std)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"segment_distances_n{n_waypoints}.png"), dpi=150)
    plt.close()
    
    # Save statistics to file
    with open(os.path.join(output_dir, f"waypoint_stats_n{n_waypoints}.pkl"), "wb") as f:
        pkl.dump(stats, f)
    
    return stats

def main():
    data_dir = "data/filtered_2k"
    analyze_waypoint_statistics(data_dir)

if __name__ == "__main__":
    main()

# # Usage example
# if __name__ == "__main__":
#     # Example usage
#     data_dir = "data/"
    
#     # Create dataloader
#     dataloader = create_trajectory_dataloader(
#         data_dir=data_dir,
#         batch_size=16,
#         n_waypoints=10,
#         image_size=(224, 224),
#         num_workers=4,
#         shuffle=True,
#         p1=0.2,  # 20% chance to use strategy 1
#         p2=0.4,  # 40% chance to use strategy 2
#         p3=0.4,  # 40% chance to use strategy 3
#     )
    
#     # Iterate through batches
#     for batch_idx, batch in enumerate(dataloader):
#         # Extract batch data
#         images = batch['image']           # Shape: [B, 3, H, W]
#         waypoints = batch['waypoints']    # Shape: [B, N, 2]
#         goals = batch['goal']             # Shape: [B, 2]
#         is_valid = batch['is_valid']      # Shape: [B]
        
#         print(f"Batch {batch_idx}:")
#         print(f"  Images shape: {images.shape}")
#         print(f"  Waypoints shape: {waypoints.shape}")
#         print(f"  Goals shape: {goals.shape}")
#         print(f"  Valid samples: {is_valid.sum().item()}/{len(is_valid)}")
        
#         # Only process a few batches for demonstration
#         if batch_idx >= 2:
#             break
            
#     print("DataLoader functioning correctly!")