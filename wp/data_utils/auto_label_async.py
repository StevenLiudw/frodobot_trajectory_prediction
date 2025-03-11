import os
import sys
import json
import time
import random
import pickle as pkl
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import logging
import asyncio
import aiohttp
from dotenv import load_dotenv
from pathlib import Path
from torch.utils.data import DataLoader
import base64
from openai import AsyncAzureOpenAI
import concurrent.futures
from functools import partial

prompt = """
Analyze this image showing a robot's view with projected waypoints. The waypoints are colored from red (nearest) to green (farthest) and represent the robot's planned path on the ground.

Please evaluate:
1. Valid: Whether the waypoints are valid. For example, if the waypoints go to or go through a wall, output False.
2. Collision: True if the path is likely to collide with BIG obstacles (walls, trees, buildings, etc.), otherwise False.
3. Off-road: True if the path is likely to go off-road, i.e. from a paved surface to a non-paved surface, otherwise False. Importantly, if the robot is currently not on a road (e.g. it's currently on grass, dirt, etc.), output False.

All waypoints are on the ground plane. Focus on whether the path is clear of obstacles and stays on the appropriate surface.

Respond in JSON format with the following structure:
{
    "explanation": "<brief explanation of your assessment>",
    "valid": <bool>,
    "collision": <bool>,
    "off_road": <bool>
}
"""

system_prompt = """
You are a computer vision expert specializing in autonomous navigation. Your task is to analyze images with projected waypoints and provide accurate assessments.
"""

# Add parent directory to path to import from wp.data_utils.dataloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wp.data_utils.dataloader import TrajectoryDataset

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncWaypointLabeler:
    """Asynchronous class to label waypoints using ChatGPT for collision and off-road detection."""
    
    def __init__(
        self,
        data_dir,
        output_dir="out/visualize_label",
        n_waypoints=10,
        max_labels=100,
        visualization_prob=0.1,
        camera_height=0.561,  # meters above ground
        camera_pitch=0,
        seed=42,
        max_retries=3,
        overwrite=False,
        shuffle=False,
        max_concurrent_requests=10,  # New parameter to control concurrency
        requests_per_minute=60  # New parameter for rate limiting
    ):
        """
        Initialize the waypoint labeler.
        
        Args:
            data_dir (str): Path to the data directory
            output_dir (str): Directory to save visualizations
            n_waypoints (int): Number of waypoints
            max_labels (int): Maximum number of samples to label
            visualization_prob (float): Probability of visualizing a sample
            camera_height (float): Height of camera above ground
            camera_pitch (float): Camera pitch in radians
            seed (int): Random seed
            max_retries (int): Maximum number of retries for API calls
            overwrite (bool): Whether to overwrite existing labels
            shuffle (bool): Whether to shuffle the dataset before labeling
            max_concurrent_requests (int): Maximum number of concurrent API requests
            requests_per_minute (int): Maximum number of API requests per minute
        """
        # Load environment variables for API keys
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
        if not self.azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")
        if not self.azure_deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT not found in environment variables")
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_waypoints = n_waypoints
        self.max_labels = max_labels
        self.visualization_prob = visualization_prob
        self.camera_height = camera_height
        self.camera_pitch = camera_pitch
        self.seed = seed
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.shuffle = shuffle
        self.max_concurrent_requests = max_concurrent_requests
        self.requests_per_minute = requests_per_minute
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Camera intrinsic matrix
        self.K = np.array([
            [203.93, 0, 192], 
            [0, 203.933, 144], 
            [0, 0, 1]
        ])
        
        # Camera distortion parameters (k1, k2, p1, p2, k3)
        self.dist_coeffs = np.array([-0.2172, 0.0537, 0.001853, -0.002105, -0.006000])
        
        # Initialize dataset
        self._init_dataset()
        
        # Initialize counters for API usage
        self.api_calls = 0
        self.tokens_used = 0
        
        # Semaphore to limit concurrent API requests
        self.semaphore = None  # Will be initialized in label_dataset
        
        # Rate limiting variables
        self.request_timestamps = []
        self.rate_limit_lock = None  # Will be initialized in label_dataset
        
    def _init_dataset(self):
        """Initialize the dataset."""
        # Define image transforms for loading the original images
        transform = None  # We'll handle the transformation manually
        
        # Set up dataset
        self.dataset = TrajectoryDataset(
            data_dir=self.data_dir,
            n_waypoints=self.n_waypoints,
            transform=transform,
            seed=self.seed,
            dummy_goal=True
        )
        
        # Shuffle the dataset samples if requested
        if self.shuffle:
            logger.info("Shuffling dataset samples")
            self.dataset.shuffle()
        
        logger.info(f"Initialized dataset with {len(self.dataset)} samples")
        
    async def _call_chatgpt_api_async(self, image_data, sample_idx, retry_count=0):
        """
        Call the Azure ChatGPT API with the given prompt and image asynchronously.
        
        Args:
            image_data (bytes): The image data in bytes
            sample_idx (int): The index of the sample (for logging)
            retry_count (int): Current retry attempt
            
        Returns:
            dict: The API response
        """
        # Apply rate limiting
        await self._apply_rate_limit()
        
        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Initialize Azure OpenAI client
        client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint
        )
        
        try:
            # Call the API
            response = await client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # Update API usage counters
            self.api_calls += 1
            
            # Record timestamp for rate limiting
            async with self.rate_limit_lock:
                self.request_timestamps.append(time.time())
            
            # Extract usage information if available
            if hasattr(response, 'usage'):
                self.tokens_used += response.usage.total_tokens
                
            # Convert the response to a dictionary
            response_dict = {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content
                        }
                    }
                ]
            }
            
            return response_dict
            
        except Exception as e:
            if retry_count < self.max_retries - 1:
                # Exponential backoff
                sleep_time = 2 ** retry_count
                logger.warning(f"API call for sample {sample_idx} failed (attempt {retry_count+1}): {e}")
                logger.info(f"Retrying in {sleep_time} seconds...")
                await asyncio.sleep(sleep_time)
                return await self._call_chatgpt_api_async(image_data, sample_idx, retry_count + 1)
            else:
                logger.error(f"Failed to call API for sample {sample_idx} after {self.max_retries} attempts: {e}")
                raise
    
    async def _apply_rate_limit(self):
        """
        Apply rate limiting to API calls based on requests_per_minute.
        Waits if necessary to stay within the rate limit.
        """
        async with self.rate_limit_lock:
            # Remove timestamps older than 1 minute
            current_time = time.time()
            one_minute_ago = current_time - 60
            self.request_timestamps = [ts for ts in self.request_timestamps if ts > one_minute_ago]
            
            # Check if we're at the rate limit
            if len(self.request_timestamps) >= self.requests_per_minute:
                # Calculate how long to wait
                oldest_timestamp = min(self.request_timestamps)
                wait_time = 60 - (current_time - oldest_timestamp)
                
                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds before next request.")
                    # Release the lock while waiting
                    self.rate_limit_lock.release()
                    try:
                        await asyncio.sleep(wait_time)
                    finally:
                        # Reacquire the lock
                        await self.rate_limit_lock.acquire()
    
    def _prepare_image_with_waypoints(self, image, waypoints):
        """
        Prepare an image with projected waypoints for visualization.
        
        Args:
            image (np.ndarray): The original image
            waypoints (np.ndarray): The waypoints to project
            
        Returns:
            np.ndarray: Image with projected waypoints
            list: List of projected waypoint coordinates
            bool: Whether the image has enough visible waypoints
        """
        # Create a copy for drawing
        image_with_waypoints = image.copy()
        
        # Convert waypoints to camera coordinates (vectorized)
        waypoints_camera = np.column_stack([
            -waypoints[:, 1],                  # x: -y_world
            np.full(len(waypoints), self.camera_height),  # y: camera_height
            waypoints[:, 0]                    # z: x_world
        ])
        
        # Filter out points behind the camera
        valid_indices = waypoints_camera[:, 2] > 0
        valid_waypoints_camera = waypoints_camera[valid_indices]
        
        # If no valid waypoints, return early
        if len(valid_waypoints_camera) == 0:
            return image_with_waypoints, [], False
        
        # Project to normalized image coordinates (before distortion)
        normalized_points = valid_waypoints_camera[:, :2] / valid_waypoints_camera[:, 2:3]
        
        # Apply distortion to normalized points
        x = normalized_points[:, 0]
        y = normalized_points[:, 1]
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r2*r4
        
        # Radial distortion
        k1, k2, p1, p2, k3 = self.dist_coeffs
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
        waypoints_image = (self.K @ distorted_points.T).T
        
        # Convert to integer coordinates
        waypoints_image_int = waypoints_image[:, :2].astype(int)
        
        # Check which points are within image bounds
        in_bounds = (
            (0 <= waypoints_image_int[:, 0]) & 
            (waypoints_image_int[:, 0] < image_with_waypoints.shape[1]) &
            (0 <= waypoints_image_int[:, 1]) & 
            (waypoints_image_int[:, 1] < image_with_waypoints.shape[0])
        )
        
        # Count visible waypoints
        visible_count = np.sum(in_bounds)
        
        # Skip if fewer than 3 waypoints are visible
        if visible_count < 3:
            return image_with_waypoints, [], False
        
        # Store all projected coordinates (even those out of bounds)
        projected_waypoints = waypoints_image_int.tolist()
        
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
        
        return image_with_waypoints, projected_waypoints, True
        
    def _parse_api_response(self, response):
        """
        Parse the API response to extract the labels.
        
        Args:
            response (dict): The API response
            
        Returns:
            dict: The parsed labels
        """
        try:
            content = response['choices'][0]['message']['content']
            
            # Parse JSON content
            labels = json.loads(content)
            
            # Validate the required fields
            required_fields = ['collision', 'off_road', 'explanation', 'valid']
            for field in required_fields:
                if field not in labels:
                    raise ValueError(f"Missing required field: {field}")
                    
            return labels
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from API response: {content}")
            raise
        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
            raise
            
    def _save_visualization(self, sample_idx, image_with_waypoints, labels, response_content):
        """
        Save a visualization of the labeled sample.
        
        Args:
            sample_idx (int): The index of the sample
            image_with_waypoints (np.ndarray): The image with projected waypoints
            labels (dict): The labels from the API
            response_content (str): The raw response content from the API
        """
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
            f"Valid: {labels['valid']}",
            f"Collision: {labels['collision']}",
            f"Off-Road: {labels['off_road']}",
        ]
        
        # Draw text with background
        y_position = 10
        for line in text_lines:
            text_width, text_height = draw.textsize(line, font=font) if hasattr(draw, 'textsize') else (100, 20)
            draw.rectangle([(10, y_position), (10 + text_width, y_position + text_height)], fill=(0, 0, 0, 128))
            draw.text((10, y_position), line, fill=(255, 255, 255), font=font)
            y_position += text_height + 5
        
        # Save the image
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        pil_image.save(os.path.join(vis_dir, f"sample_{sample_idx}_labeled.jpg"))
        
        # Save the full API response
        response_dir = os.path.join(self.output_dir, "responses")
        os.makedirs(response_dir, exist_ok=True)
        with open(os.path.join(response_dir, f"sample_{sample_idx}_response.json"), "w") as f:
            f.write(response_content)
            
    def _save_insufficient_waypoints_label(self, label_path):
        """
        Save a special label for samples with insufficient visible waypoints.
        
        Args:
            label_path (str): Path to save the label
        """
        # Create a special label for insufficient waypoints
        insufficient_label = {
            "explanation": "Insufficient visible waypoints",
            "valid": False,
            "collision": False,
            "off_road": False,
            "insufficient_waypoints": True
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        # Save the label
        with open(label_path, "wb") as f:
            pkl.dump(insufficient_label, f)
            
        logger.debug(f"Saved insufficient waypoints label to {label_path}")
    
    async def _process_sample(self, sample_idx):
        """
        Process a single sample asynchronously.
        
        Args:
            sample_idx (int): The index of the sample
            
        Returns:
            dict: Processing result with status and details
        """
        # Get the sample info
        sample_info = self.dataset.samples[sample_idx]
        ride_dir = sample_info["ride_dir"]
        img_idx = sample_info["img_idx"]
        
        # Check if label already exists
        label_path = os.path.join(ride_dir, "labels", f"{img_idx}.pkl")
        
        # Skip if label exists and we're not overwriting
        if os.path.exists(label_path) and not self.overwrite:
            return {
                "status": "skipped",
                "reason": "label_exists",
                "sample_idx": sample_idx
            }
            
        try:
            # Get the sample
            sample = self.dataset[sample_idx]
            
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
            image_with_waypoints, projected_waypoints, has_enough_waypoints = self._prepare_image_with_waypoints(image, waypoints)
            
            # If not enough visible waypoints, save a special label and continue
            if not has_enough_waypoints:
                self._save_insufficient_waypoints_label(label_path)
                return {
                    "status": "skipped",
                    "reason": "insufficient_waypoints",
                    "sample_idx": sample_idx
                }
            
            # Convert image to bytes for API
            _, img_encoded = cv2.imencode('.jpg', image_with_waypoints)
            img_bytes = img_encoded.tobytes()
            
            # Acquire semaphore to limit concurrent requests
            async with self.semaphore:
                # Call ChatGPT API
                logger.debug(f"Calling ChatGPT API for sample {sample_idx}")
                response = await self._call_chatgpt_api_async(img_bytes, sample_idx)
            
            # Parse the response
            labels = self._parse_api_response(response)
            
            # Save the labels
            os.makedirs(os.path.join(ride_dir, "labels"), exist_ok=True)
            with open(label_path, "wb") as f:
                pkl.dump(labels, f)
            
            # Visualize with some probability
            if random.random() < self.visualization_prob:
                self._save_visualization(
                    sample_idx, 
                    image_with_waypoints, 
                    labels, 
                    response['choices'][0]['message']['content']
                )
            
            return {
                "status": "success",
                "sample_idx": sample_idx,
                "labels": labels
            }
                
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            return {
                "status": "error",
                "sample_idx": sample_idx,
                "error": str(e)
            }
    
    async def label_dataset(self):
        """
        Label the dataset using ChatGPT asynchronously.
        
        Returns:
            dict: Statistics about the labeling process
        """
        logger.info(f"Starting to label dataset with max {self.max_labels} samples")
        
        # Initialize counters
        labeled_count = 0
        skipped_count = 0
        error_count = 0
        insufficient_waypoints_count = 0
        
        # Initialize semaphore for limiting concurrent requests
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Initialize rate limiting lock
        self.rate_limit_lock = asyncio.Lock()
        self.request_timestamps = []
        
        # Process samples in batches
        batch_size = 100  # Process 100 samples at a time
        total_samples = min(len(self.dataset), self.max_labels)
        
        # Create a progress bar for the entire process
        pbar = tqdm(total=total_samples)
        
        # Process in batches
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            logger.info(f"Processing batch {batch_start} to {batch_end-1}")
            
            # Create tasks for this batch
            batch_tasks = []
            for sample_idx in range(batch_start, batch_end):
                batch_tasks.append(self._process_sample(sample_idx))
            
            # Process batch tasks
            batch_results = []
            for task in asyncio.as_completed(batch_tasks):
                result = await task
                batch_results.append(result)
                
                # Update counters based on result status
                if result["status"] == "success":
                    labeled_count += 1
                elif result["status"] == "skipped":
                    skipped_count += 1
                    if result["reason"] == "insufficient_waypoints":
                        insufficient_waypoints_count += 1
                elif result["status"] == "error":
                    error_count += 1
                
                # Update progress bar
                pbar.update(1)
            
            # Log progress after each batch
            logger.info(f"Completed batch. Total processed: {batch_end}, labeled {labeled_count}, skipped {skipped_count}, errors {error_count}")
        
        pbar.close()
        
        # Log final statistics
        logger.info(f"Labeling complete. Labeled {labeled_count} samples, skipped {skipped_count}, errors {error_count}")
        logger.info(f"Skipped {insufficient_waypoints_count} samples due to insufficient visible waypoints")
        logger.info(f"API usage: {self.api_calls} calls, {self.tokens_used} tokens")
        
        return {
            "labeled_count": labeled_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "insufficient_waypoints_count": insufficient_waypoints_count,
            "api_calls": self.api_calls,
            "tokens_used": self.tokens_used
        }

async def main_async():
    """Async main function to run the labeler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Label waypoints using ChatGPT (async version)")
    parser.add_argument("--data_dir", type=str, default="data/filtered_2k", help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="out/visualize_label", help="Directory to save visualizations")
    parser.add_argument("--n_waypoints", type=int, default=10, help="Number of waypoints")
    parser.add_argument("--max_labels", type=int, default=100, help="Maximum number of samples to label")
    parser.add_argument("--visualization_prob", type=float, default=0.1, help="Probability of visualizing a sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset before labeling")
    parser.add_argument("--max_concurrent", type=int, default=100, help="Maximum number of concurrent API requests")
    parser.add_argument("--rpm", type=int, default=400, help="Maximum requests per minute (rate limit)")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of threads to use for processing")
    
    args = parser.parse_args()
    
    # Create the labeler
    labeler = AsyncWaypointLabeler(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_waypoints=args.n_waypoints,
        max_labels=args.max_labels,
        visualization_prob=args.visualization_prob,
        seed=args.seed,
        overwrite=args.overwrite,
        shuffle=args.shuffle,
        max_concurrent_requests=args.max_concurrent,
        requests_per_minute=args.rpm
    )
    
    # Run the labeling process with multiple threads
    stats = await run_multithreaded(labeler, args.num_threads)
    
    # Print final statistics
    print("\nLabeling Statistics:")
    print(f"Labeled samples: {stats['labeled_count']}")
    print(f"Skipped samples: {stats['skipped_count']}")
    print(f"Insufficient waypoints: {stats['insufficient_waypoints_count']}")
    print(f"Errors: {stats['error_count']}")
    print(f"API calls: {stats['api_calls']}")
    print(f"Tokens used: {stats['tokens_used']}")

async def process_subset(labeler, sample_indices):
    """Process a subset of samples in the dataset.
    
    Args:
        labeler: The AsyncWaypointLabeler instance
        sample_indices: List of sample indices to process
        
    Returns:
        dict: Statistics about the labeling process for this subset
    """
    # Initialize counters
    labeled_count = 0
    skipped_count = 0
    error_count = 0
    insufficient_waypoints_count = 0
    api_calls = 0
    tokens_used = 0
    
    # Initialize semaphore for limiting concurrent requests
    labeler.semaphore = asyncio.Semaphore(labeler.max_concurrent_requests)
    
    # Initialize rate limiting lock
    labeler.rate_limit_lock = asyncio.Lock()
    labeler.request_timestamps = []
    
    # Process samples in batches
    batch_size = 100  # Process 100 samples at a time
    total_samples = len(sample_indices)
    
    # Create a progress bar
    pbar = tqdm(total=total_samples, position=0, leave=True)
    
    # Process in batches
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        logger.info(f"Processing batch {batch_start} to {batch_end-1} in thread")
        
        # Create tasks for this batch
        batch_tasks = []
        for i in range(batch_start, batch_end):
            sample_idx = sample_indices[i]
            batch_tasks.append(labeler._process_sample(sample_idx))
        
        # Process batch tasks
        for task in asyncio.as_completed(batch_tasks):
            result = await task
            
            # Update counters based on result status
            if result["status"] == "success":
                labeled_count += 1
            elif result["status"] == "skipped":
                skipped_count += 1
                if result["reason"] == "insufficient_waypoints":
                    insufficient_waypoints_count += 1
            elif result["status"] == "error":
                error_count += 1
            
            # Update progress bar
            pbar.update(1)
    
    pbar.close()
    
    # Get API usage from labeler
    api_calls = labeler.api_calls
    tokens_used = labeler.tokens_used
    
    return {
        "labeled_count": labeled_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "insufficient_waypoints_count": insufficient_waypoints_count,
        "api_calls": api_calls,
        "tokens_used": tokens_used
    }

def thread_worker(labeler, sample_indices):
    """Worker function for each thread.
    
    Args:
        labeler: The AsyncWaypointLabeler instance
        sample_indices: List of sample indices to process
        
    Returns:
        dict: Statistics about the labeling process for this subset
    """
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the async function in this thread's event loop
    try:
        return loop.run_until_complete(process_subset(labeler, sample_indices))
    finally:
        loop.close()

async def run_multithreaded(labeler, num_threads):
    """Run the labeling process with multiple threads.
    
    Args:
        labeler: The AsyncWaypointLabeler instance
        num_threads: Number of threads to use
        
    Returns:
        dict: Combined statistics from all threads
    """
    # Determine the total number of samples to process
    total_dataset_size = len(labeler.dataset)
    samples_to_process = min(total_dataset_size, labeler.max_labels)
    
    # Create sample indices
    if labeler.shuffle:
        # If shuffle is enabled, randomly sample from the entire dataset
        all_indices = list(range(total_dataset_size))
        random.shuffle(all_indices)
        sample_indices = all_indices[:samples_to_process]
    else:
        # If not shuffling, just take the first samples_to_process samples
        sample_indices = list(range(samples_to_process))
    
    # Divide indices among threads
    subset_size = samples_to_process // num_threads
    subsets = []
    
    for i in range(num_threads):
        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size if i < num_threads - 1 else samples_to_process
        subsets.append(sample_indices[start_idx:end_idx])
    
    logger.info(f"Splitting {samples_to_process} samples across {num_threads} threads")
    
    # Create a thread pool and submit tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create a separate labeler instance for each thread to avoid shared state issues
        thread_labelers = []
        for i in range(num_threads):
            thread_labeler = AsyncWaypointLabeler(
                data_dir=labeler.data_dir,
                output_dir=labeler.output_dir,
                n_waypoints=labeler.n_waypoints,
                max_labels=labeler.max_labels,
                visualization_prob=labeler.visualization_prob,
                camera_height=labeler.camera_height,
                camera_pitch=labeler.camera_pitch,
                seed=labeler.seed + i,  # Use different seeds for each thread
                max_retries=labeler.max_retries,
                overwrite=labeler.overwrite,
                shuffle=False,  # Already shuffled in the main labeler
                max_concurrent_requests=labeler.max_concurrent_requests // num_threads,  # Divide concurrency
                requests_per_minute=labeler.requests_per_minute // num_threads  # Divide rate limit
            )
            thread_labelers.append(thread_labeler)
        
        # Submit tasks to the thread pool
        futures = []
        for i in range(num_threads):
            futures.append(executor.submit(thread_worker, thread_labelers[i], subsets[i]))
        
        # Wait for all futures to complete and collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Thread error: {e}")
    
    # Combine statistics from all threads
    combined_stats = {
        "labeled_count": sum(r["labeled_count"] for r in results),
        "skipped_count": sum(r["skipped_count"] for r in results),
        "error_count": sum(r["error_count"] for r in results),
        "insufficient_waypoints_count": sum(r["insufficient_waypoints_count"] for r in results),
        "api_calls": sum(r["api_calls"] for r in results),
        "tokens_used": sum(r["tokens_used"] for r in results)
    }
    
    return combined_stats

def main():
    """Entry point for the script."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
