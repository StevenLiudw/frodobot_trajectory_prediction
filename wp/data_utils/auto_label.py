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
import requests
from dotenv import load_dotenv
from pathlib import Path
from torch.utils.data import DataLoader
import base64
from openai import AzureOpenAI

prompt = """
Analyze this image showing a robot's view with projected waypoints. The waypoints are colored from red (nearest) to green (farthest) and represent the robot's planned path on the ground.

Please evaluate:
1. Valid: Whether the waypoints are valid. For example, if the waypoints go to or go through a wall, output False.
2. Collision: True if the path is likely to collide with obstacles (walls, trees, buildings, etc.), otherwise False.
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WaypointLabeler:
    """Class to label waypoints using ChatGPT for collision and off-road detection."""
    
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
        shuffle=False,  # New parameter to control dataset shuffling
        manual_mode=False,  # New parameter for manual labeling
        use_batch=False,
        batch_size=3500
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
            manual_mode (bool): Whether to use manual labeling mode
        """
        # Load environment variables for API keys
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if use_batch:
            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_BATCH")
        else:
            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if not self.api_key and not manual_mode:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
        if not self.azure_endpoint and not manual_mode:
            raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")
        if not self.azure_deployment and not manual_mode:
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
        self.shuffle = shuffle  # Store the shuffle parameter
        self.manual_mode = manual_mode  # Store the manual mode parameter
        self.use_batch = use_batch
        self.batch_size = batch_size
        
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

         # Create directories for batch processing
        if self.use_batch:
            self.batch_dir = os.path.join(self.output_dir, "batch")
            self.batch_results_dir = os.path.join(self.output_dir, "batch_results")
            os.makedirs(self.batch_dir, exist_ok=True)
            os.makedirs(self.batch_results_dir, exist_ok=True)
        
    def _init_dataset(self):
        """Initialize the dataset."""
        # Define image transforms for loading the original images
        transform = None  # We'll handle the transformation manually
        
        # Set up dataset
        self.dataset: TrajectoryDataset = TrajectoryDataset(
            data_dir=self.data_dir,
            n_waypoints=self.n_waypoints,
            transform=transform,
            seed=self.seed
        )
        
        # Shuffle the dataset samples if requested
        if self.shuffle:
            logger.info("Shuffling dataset samples")
            random.shuffle(self.dataset.samples)
        
        logger.info(f"Initialized dataset with {len(self.dataset)} samples")
        
    def _call_chatgpt_api(self, image_data):
        """
        Call the Azure ChatGPT API with the given prompt and image.
        
        Args:
            image_data (bytes): The image data in bytes
            
        Returns:
            dict: The API response
        """
        
        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.azure_api_version,  # Update this to the appropriate API version
            azure_endpoint=self.azure_endpoint
        )
        
        for attempt in range(self.max_retries):
            try:
                # Call the API
                response = client.chat.completions.create(
                    model=self.azure_deployment,  # Use the deployment name instead of model name
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
                logger.warning(f"API call attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to call API after {self.max_retries} attempts")
                    raise
        
    def _prepare_batch_file(self, batch_samples):
        """
        Prepare a batch file for the Azure OpenAI batch API.
        
        Args:
            batch_samples (list): List of dictionaries containing sample information
            
        Returns:
            str: Path to the batch file
            list: List of sample identifiers in the batch
        """
        batch_data = []
        sample_identifiers = []
        
        for sample_info in batch_samples:
            sample_idx = sample_info["sample_idx"]
            image_data = sample_info["image_data"]
            ride_dir = sample_info["ride_dir"]
            img_idx = sample_info["img_idx"]
            
            # Create a stable identifier that doesn't depend on dataset order
            # Format: "ride_dir:img_idx" (using the last part of ride_dir for brevity)
            ride_name = os.path.basename(os.path.normpath(ride_dir))
            stable_id = f"{ride_name}:{img_idx}"
            
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create the batch entry in the correct format
            batch_entry = {
                "custom_id": stable_id,
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self.azure_deployment,
                    "messages": [
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
                    "response_format": {"type": "json_object"}
                }
            }
            
            batch_data.append(batch_entry)
            # Store both the sample_idx and the stable_id for reference
            sample_identifiers.append({
                "sample_idx": sample_idx,
                "stable_id": stable_id,
                "ride_dir": ride_dir,
                "img_idx": img_idx
            })
        
        # Create a unique batch file name
        timestamp = int(time.time())
        batch_file_path = os.path.join(self.batch_dir, f"batch_{timestamp}.jsonl")
        
        # Write the batch file in JSONL format
        with open(batch_file_path, "w") as f:
            for entry in batch_data:
                f.write(json.dumps(entry) + "\n")
        
        logger.info(f"Created batch file with {len(batch_data)} samples at {batch_file_path}")
        
        return batch_file_path, sample_identifiers
        
    def _submit_batch(self, batch_file_path):
        """
        Submit a batch file to the Azure OpenAI batch API.
        
        Args:
            batch_file_path (str): Path to the batch file
            
        Returns:
            str: Batch ID
        """
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint
        )
        
        # Submit the batch
        for attempt in range(self.max_retries):
            try:
                # First upload the file with purpose="batch"
                with open(batch_file_path, "rb") as f:
                    file_response = client.files.create(
                        file=f,
                        purpose="batch"
                    )
                
                file_id = file_response.id
                logger.info(f"Uploaded batch file with ID: {file_id}")
                
                # Create a batch job with the file
                batch_response = client.batches.create(
                    input_file_id=file_id,
                    endpoint="/chat/completions",
                    completion_window="24h"
                )
                
                # Update API usage counters
                self.api_calls += 1
                
                logger.info(f"Submitted batch with ID: {batch_response.id}")
                
                return batch_response.id
                
            except Exception as e:
                logger.warning(f"Batch submission attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to submit batch after {self.max_retries} attempts")
                    raise
    
    def _check_batch_status(self, batch_id):
        """
        Check the status of a batch.
        
        Args:
            batch_id (str): Batch ID
            
        Returns:
            str: Batch status
        """
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint
        )
        
        # Get batch status
        for attempt in range(self.max_retries):
            try:
                batch_status = client.batches.retrieve(batch_id)
                return batch_status.status
                
            except Exception as e:
                logger.warning(f"Batch status check attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to check batch status after {self.max_retries} attempts")
                    raise
    
    def _fetch_batch_results(self, batch_id):
        """
        Fetch the results of a completed batch.
        
        Args:
            batch_id (str): Batch ID
            
        Returns:
            list: List of batch results
        """
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint
        )
        
        # Get batch results
        for attempt in range(self.max_retries):
            try:
                # Get the batch info
                batch_info = client.batches.retrieve(batch_id)
                
                # Check if the batch is completed
                if batch_info.status != "completed":
                    logger.warning(f"Batch {batch_id} is not completed yet. Status: {batch_info.status}")
                    if attempt < self.max_retries - 1:
                        # Wait before retrying
                        sleep_time = 60  # Wait a minute before checking again
                        logger.info(f"Waiting {sleep_time} seconds before checking again...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise ValueError(f"Batch {batch_id} did not complete in time")
                
                # Get the output file ID
                output_file_id = batch_info.output_file_id
                if not output_file_id:
                    if batch_info.error_file_id:
                        # If there's an error file, download it to see the errors
                        error_file_id = batch_info.error_file_id
                        error_content = client.files.content(error_file_id)
                        error_path = os.path.join(self.batch_results_dir, f"error_{batch_id}.jsonl")
                        
                        with open(error_path, "wb") as f:
                            f.write(error_content.content)
                        
                        logger.error(f"Batch {batch_id} had errors. Error file saved to {error_path}")
                    
                    raise ValueError(f"No output file found for batch {batch_id}")
                
                # Download the output file
                output_content = client.files.content(output_file_id)
                results_file_path = os.path.join(self.batch_results_dir, f"results_{batch_id}.jsonl")
                
                # Save the raw content
                with open(results_file_path, "wb") as f:
                    f.write(output_content.content)
                
                logger.info(f"Downloaded batch results to {results_file_path}")
                
                # Parse the results
                results = []
                with open(results_file_path, "r") as f:
                    for line in f:
                        results.append(json.loads(line))
                
                return results
                
            except Exception as e:
                logger.warning(f"Batch results fetch attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to fetch batch results after {self.max_retries} attempts")
                    raise
    
    def _process_batch_results(self, batch_results, sample_identifiers):
        """
        Process batch results and save labels.
        
        Args:
            batch_results (list): List of batch results
            sample_identifiers (list): List of sample identifier dictionaries
            
        Returns:
            int: Number of successfully processed samples
        """
        processed_count = 0
        
        # Create a mapping from custom_id to result
        results_map = {}
        for result in batch_results:
            if 'custom_id' in result:
                custom_id = result['custom_id']
                results_map[custom_id] = result
        
        # Process each sample
        for sample_info in sample_identifiers:
            stable_id = sample_info["stable_id"]
            ride_dir = sample_info["ride_dir"]
            img_idx = sample_info["img_idx"]
            sample_idx = sample_info["sample_idx"]  # Original sample_idx, only used for logging
            
            if stable_id not in results_map:
                logger.warning(f"No result found for sample {stable_id}")
                continue
            
            result = results_map[stable_id]
            
            try:
                # Check if there was an error
                if result.get('error'):
                    logger.error(f"Error in batch result for sample {stable_id}: {result['error']}")
                    continue
                
                # Parse the response
                if 'response' in result and 'body' in result['response'] and 'choices' in result['response']['body']:
                    choices = result['response']['body']['choices']
                    if len(choices) > 0 and 'message' in choices[0]:
                        content = choices[0]['message']['content']
                        labels = json.loads(content)
                        
                        # Save the labels
                        label_path = os.path.join(ride_dir, "labels", f"{img_idx}.pkl")
                        os.makedirs(os.path.join(ride_dir, "labels"), exist_ok=True)
                        with open(label_path, "wb") as f:
                            pkl.dump(labels, f)
                        
                        processed_count += 1
                        
                        # Visualize with some probability
                        if random.random() < self.visualization_prob:
                            # We need to find this sample in the current dataset
                            # This is tricky if the dataset was reshuffled, so we'll load it directly
                            try:
                                # Load the image directly from the file
                                img_path = os.path.join(ride_dir, "images", f"{img_idx}.jpg")
                                if os.path.exists(img_path):
                                    image = cv2.imread(img_path)
                                    
                                    # Get waypoints from the dataset
                                    waypoints_path = os.path.join(ride_dir, "waypoints", f"{img_idx}.npy")
                                    if os.path.exists(waypoints_path):
                                        waypoints = np.load(waypoints_path)
                                        
                                        # Prepare image with waypoints
                                        image_with_waypoints, _, _ = self._prepare_image_with_waypoints(image, waypoints)
                                        
                                        # Save visualization
                                        self._save_visualization(
                                            f"{stable_id}",  # Use stable_id instead of sample_idx
                                            image_with_waypoints,
                                            labels,
                                            content
                                        )
                            except Exception as viz_error:
                                logger.warning(f"Error creating visualization for {stable_id}: {viz_error}")
                    else:
                        logger.warning(f"No choices found in response for sample {stable_id}")
                else:
                    logger.warning(f"Invalid result format for sample {stable_id}")
            
            except Exception as e:
                logger.error(f"Error processing batch result for sample {stable_id}: {e}")
        
        return processed_count
    
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
            required_fields = ['collision', 'off_road', 'explanation']
            for field in required_fields:
                if field not in labels:
                    raise ValueError(f"Missing required field: {field}")
                    
            # # Validate the values
            # if not (0 <= labels['collision'] <= 1):
            #     raise ValueError(f"collision must be between 0 and 1, got {labels['collision']}")
                
            # if not (0 <= labels['off_road'] <= 1):
            #     raise ValueError(f"off_road must be between 0 and 1, got {labels['off_road']}")
                
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
            # f"Explanation: {labels['explanation'][:50]}..." if len(labels['explanation']) > 50 else f"Explanation: {labels['explanation']}"
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
            "explanation": "Insufficient visible waypoints in image",
            "collision_risk": -1.0,  # Special value to indicate insufficient waypoints
            "off_road_risk": -1.0,   # Special value to indicate insufficient waypoints
            "insufficient_waypoints": True
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        # Save the label
        with open(label_path, "wb") as f:
            pkl.dump(insufficient_label, f)
            
        logger.debug(f"Saved insufficient waypoints label to {label_path}")
            
    def _manual_label(self, sample_idx, image_with_waypoints):
        """
        Manually label a sample by showing the image and waiting for user input.
        
        Args:
            sample_idx (int): The index of the sample
            image_with_waypoints (np.ndarray): The image with projected waypoints
            
        Returns:
            dict: The manual labels
        """
        # Fix color issue (OpenCV uses BGR, but we need RGB for display)
        image_with_waypoints_rgb = cv2.cvtColor(image_with_waypoints, cv2.COLOR_BGR2RGB)
        
        # Create a window for displaying the image
        window_name = f"Sample {sample_idx} - Press a key to label"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        # # Add instructions to the image
        # instructions = [
        #     "a: Very Bad (0) | s: Bad (1) | k: Good (2) | l: Very Good (3) | ESC: Quit"
        # ]
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # y_pos = 30
        # for line in instructions:
        #     cv2.putText(image_with_waypoints_rgb, line, (10, y_pos), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        #     y_pos += 30
            
        cv2.imshow(window_name, image_with_waypoints_rgb)
        
        # Wait for user input
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            # a key (97) - Very Bad (0)
            if key == 97:  # 'a'
                cv2.destroyWindow(window_name)
                return {
                    "explanation": "Manually labeled as very bad (0)",
                    "valid": False,
                    "collision": True,
                    "off_road": True,
                    "quality": 0
                }
            
            # s key (115) - Bad (1)
            elif key == 115:  # 's'
                cv2.destroyWindow(window_name)
                return {
                    "explanation": "Manually labeled as bad (1)",
                    "valid": False,
                    "collision": True,
                    "off_road": False,
                    "quality": 1
                }
            
            # k key (107) - Good (2)
            elif key == 107:  # 'k'
                cv2.destroyWindow(window_name)
                return {
                    "explanation": "Manually labeled as good (2)",
                    "valid": True,
                    "collision": False,
                    "off_road": False,
                    "quality": 2
                }
            
            # l key (108) - Very Good (3)
            elif key == 108:  # 'l'
                cv2.destroyWindow(window_name)
                return {
                    "explanation": "Manually labeled as very good (3)",
                    "valid": True,
                    "collision": False,
                    "off_road": False,
                    "quality": 3
                }
            
            # ESC key (27) - Quit
            elif key == 27:
                cv2.destroyWindow(window_name)
                return None
    
    def label_dataset(self):
        """
        Label the dataset using ChatGPT or manual labeling.
        
        Returns:
            dict: Statistics about the labeling process
        """
        logger.info(f"Starting to label dataset with max {self.max_labels} samples")
        
        # Initialize counters
        labeled_count = 0
        skipped_count = 0
        error_count = 0
        insufficient_waypoints_count = 0
        submitted_count = 0
        
        # Create a progress bar
        pbar = tqdm(total=min(self.max_labels, len(self.dataset)))
        
        if self.use_batch and not self.manual_mode:
            # Batch processing mode
            logger.info("Using batch processing mode")
            
            # Collect samples for batch processing
            batch_samples = []
            submitted_batch_ids = []
            
            # First pass: collect valid samples for batch processing
            for sample_idx in range(len(self.dataset)):
                # Check if we've reached the maximum number of labels
                if submitted_count + len(batch_samples) >= self.max_labels:
                    logger.info(f"Reached maximum number of labels ({self.max_labels})")
                    break
                
                # Get the sample info
                sample_info = self.dataset.samples[sample_idx]
                ride_dir = sample_info["ride_dir"]
                img_idx = sample_info["img_idx"]
                
                # Check if label already exists
                label_path = os.path.join(ride_dir, "labels", f"{img_idx}.pkl")
                
                # Skip if label exists and we're not overwriting
                if os.path.exists(label_path) and not self.overwrite:
                    logger.debug(f"Label already exists for sample {sample_idx}, skipping")
                    skipped_count += 1
                    continue
                
                try:
                    # Get the sample
                    sample = self.dataset[sample_idx]
                    
                    # Skip invalid samples
                    if not sample.get('is_valid', True):
                        logger.debug(f"Sample {sample_idx} is invalid, skipping")
                        skipped_count += 1
                        continue
                    
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
                        logger.debug(f"Sample {sample_idx} has fewer than 3 visible waypoints")
                        self._save_insufficient_waypoints_label(label_path)
                        insufficient_waypoints_count += 1
                        skipped_count += 1
                        pbar.update(1)
                        continue
                    
                    # Convert image to bytes for API
                    _, img_encoded = cv2.imencode('.jpg', image_with_waypoints)
                    img_bytes = img_encoded.tobytes()
                    
                    # Add to batch samples with ride_dir and img_idx for stable identification
                    batch_samples.append({
                        "sample_idx": sample_idx,
                        "image_data": img_bytes,
                        "ride_dir": ride_dir,
                        "img_idx": img_idx
                    })
                    
                    # If we have enough samples for a batch, submit it (but don't wait)
                    if len(batch_samples) >= self.batch_size:
                        batch_file_path, sample_identifiers = self._prepare_batch_file(batch_samples)
                        batch_id = self._submit_batch(batch_file_path)
                        
                        # Save batch ID and sample identifiers for later retrieval
                        batch_info = {
                            "batch_id": batch_id,
                            "sample_identifiers": sample_identifiers,
                            "timestamp": time.time(),
                            "status": "submitted"
                        }
                        
                        # Save batch info to a file
                        batch_info_path = os.path.join(self.batch_dir, f"batch_info_{batch_id}.json")
                        with open(batch_info_path, "w") as f:
                            json.dump(batch_info, f)
                        
                        submitted_batch_ids.append(batch_id)
                        logger.info(f"Submitted batch {batch_id} with {len(sample_identifiers)} samples")
                        
                        # Clear batch samples
                        batch_samples = []
                        submitted_count += len(sample_identifiers)
                except Exception as e:
                    logger.error(f"Error processing sample {sample_idx}: {e}")
                    error_count += 1
                    
                    # If too many errors, stop
                    if error_count > 10:
                        logger.error("Too many errors, stopping")
                        break
            
            # Submit any remaining samples
            if batch_samples:
                batch_file_path, sample_identifiers = self._prepare_batch_file(batch_samples)
                batch_id = self._submit_batch(batch_file_path)
                
                # Save batch ID and sample identifiers for later retrieval
                batch_info = {
                    "batch_id": batch_id,
                    "sample_identifiers": sample_identifiers,
                    "timestamp": time.time(),
                    "status": "submitted"
                }
                
                # Save batch info to a file
                batch_info_path = os.path.join(self.batch_dir, f"batch_info_{batch_id}.json")
                with open(batch_info_path, "w") as f:
                    json.dump(batch_info, f)
                
                submitted_batch_ids.append(batch_id)
                logger.info(f"Submitted batch {batch_id} with {len(sample_identifiers)} samples")
            
            # Save all submitted batch IDs to a summary file
            batch_summary = {
                "submitted_batch_ids": submitted_batch_ids,
                "timestamp": time.time()
            }
            
            batch_summary_path = os.path.join(self.batch_dir, f"batch_summary_{int(time.time())}.json")
            with open(batch_summary_path, "w") as f:
                json.dump(batch_summary, f)
            
            logger.info(f"Submitted {len(submitted_batch_ids)} batches in total. Batch IDs saved to {batch_summary_path}")
            logger.info("Use fetch_batch_results method later to retrieve and process the results")
            
            # Update progress bar to show submission is complete
            pbar.update(min(self.max_labels, len(self.dataset)) - pbar.n)
        else:
            # Original single-sample processing mode
            # Iterate through the dataset
            for sample_idx in range(len(self.dataset)):
                # Check if we've reached the maximum number of labels
                if labeled_count >= self.max_labels:
                    logger.info(f"Reached maximum number of labels ({self.max_labels})")
                    break
                    
                # Get the sample info
                sample_info = self.dataset.samples[sample_idx]
                ride_dir = sample_info["ride_dir"]
                img_idx = sample_info["img_idx"]
                
                # Check if label already exists
                label_path = os.path.join(ride_dir, "labels", f"{img_idx}.pkl")
                
                # Skip if label exists and we're not overwriting
                if os.path.exists(label_path) and not self.overwrite:
                    logger.debug(f"Label already exists for sample {sample_idx}, skipping")
                    skipped_count += 1
                    continue
                    
                try:
                    # Get the sample
                    sample = self.dataset[sample_idx]
                    
                    # Skip invalid samples
                    if not sample.get('is_valid', True):
                        logger.debug(f"Sample {sample_idx} is invalid, skipping")
                        skipped_count += 1
                        continue
                    
                    # Get image and waypoints
                    image = sample['image']  # Shape: [3, H, W]
                    waypoints = sample['waypoints']  # Shape: [N, 2]
                    
                    # Convert image tensor to numpy for visualization
                    if isinstance(image, torch.Tensor):
                        # If normalized, unnormalize
                        if image.max() <= 1.0:
                            image = image.permute(1, 2, 0).numpy()  # [H, W, 3]
                        else:
                            image = image.permute(1, 2, 0).numpy() / 255.0  # [H, W, 3]
                        image = (image * 255).astype(np.uint8)  # Scale to 0-255
                    
                    # Prepare image with waypoints
                    image_with_waypoints, projected_waypoints, has_enough_waypoints = self._prepare_image_with_waypoints(image, waypoints)
                    
                    # If not enough visible waypoints, save a special label and continue
                    if not has_enough_waypoints:
                        logger.debug(f"Sample {sample_idx} has fewer than 3 visible waypoints")
                        self._save_insufficient_waypoints_label(label_path)
                        insufficient_waypoints_count += 1
                        skipped_count += 1
                        pbar.update(1)
                        continue
                    
                    # Handle manual or automatic labeling
                    if self.manual_mode:
                        # Manual labeling
                        labels = self._manual_label(sample_idx, image_with_waypoints)
                        
                        # If user chose to quit
                        if labels is None:
                            logger.info("Manual labeling stopped by user")
                            break
                    else:
                        # Automatic labeling with API
                        # Convert image to bytes for API
                        _, img_encoded = cv2.imencode('.jpg', image_with_waypoints)
                        img_bytes = img_encoded.tobytes()
                        
                        # Call ChatGPT API
                        response = self._call_chatgpt_api(img_bytes)
                        
                        # Parse the response
                        labels = self._parse_api_response(response)
                    
                    # Save the labels
                    os.makedirs(os.path.join(ride_dir, "labels"), exist_ok=True)
                    with open(label_path, "wb") as f:
                        pkl.dump(labels, f)
                    
                    # Visualize with some probability (only for automatic mode)
                    if random.random() < self.visualization_prob:
                        self._save_visualization(
                            sample_idx, 
                            image_with_waypoints, 
                            labels, 
                            response['choices'][0]['message']['content'] if not self.manual_mode else "Manual labeling"
                        )
                    
                    labeled_count += 1
                    pbar.update(1)
                    
                    # Log progress
                    if labeled_count % 10 == 0:
                        if self.manual_mode:
                            logger.info(f"Manually labeled {labeled_count} samples")
                        else:
                            logger.info(f"Labeled {labeled_count} samples, API calls: {self.api_calls}, tokens used: {self.tokens_used}")
                    
                except Exception as e:
                    logger.error(f"Error processing sample {sample_idx}: {e}")
                    error_count += 1
                    
                    # If too many errors, stop
                    if error_count > 10:
                        logger.error("Too many errors, stopping")
                        break
        
        pbar.close()
        
        # Log final statistics
        logger.info(f"Labeling complete. Labeled {labeled_count} samples, skipped {skipped_count}, errors {error_count}")
        logger.info(f"Skipped {insufficient_waypoints_count} samples due to insufficient visible waypoints")
        if not self.manual_mode:
            logger.info(f"API usage: {self.api_calls} calls, {self.tokens_used} tokens")
        
        return {
            "labeled_count": labeled_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "insufficient_waypoints_count": insufficient_waypoints_count,
            "api_calls": self.api_calls if not self.manual_mode else 0,
            "tokens_used": self.tokens_used if not self.manual_mode else 0
        }

    def fetch_batch_results(self, batch_id=None):
        """
        Fetch and process results from previously submitted batches.
        
        Args:
            batch_id (str, optional): Specific batch ID to fetch. If None, try to fetch all pending batches.
            
        Returns:
            dict: Statistics about the processing
        """
        logger.info("Starting to fetch and process batch results")
        
        # Initialize counters
        processed_count = 0
        error_count = 0
        
        # Directory for batch info
        batch_dir = os.path.join(self.output_dir, "batch")
        
        # If a specific batch ID is provided
        if batch_id:
            batch_info_path = os.path.join(batch_dir, f"batch_info_{batch_id}.json")
            if not os.path.exists(batch_info_path):
                logger.error(f"Batch info file not found for batch ID {batch_id}")
                return {"processed_count": 0, "error_count": 1}
            
            with open(batch_info_path, "r") as f:
                batch_info = json.load(f)
            
            # Check batch status
            status = self._check_batch_status(batch_id)
            logger.info(f"Batch {batch_id} status: {status}")
            
            if status == "succeeded":
                try:
                    # Fetch and process results
                    batch_results = self._fetch_batch_results(batch_id)
                    processed = self._process_batch_results(batch_results, batch_info["sample_identifiers"])
                    
                    # Update batch info
                    batch_info["status"] = "processed"
                    batch_info["processed_timestamp"] = time.time()
                    batch_info["processed_count"] = processed
                    
                    with open(batch_info_path, "w") as f:
                        json.dump(batch_info, f)
                    
                    processed_count += processed
                    logger.info(f"Processed {processed} samples from batch {batch_id}")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_id}: {e}")
                    error_count += 1
                    
                    # Update batch info
                    batch_info["status"] = "error"
                    batch_info["error_timestamp"] = time.time()
                    batch_info["error_message"] = str(e)
                    
                    with open(batch_info_path, "w") as f:
                        json.dump(batch_info, f)
            elif status in ["failed", "cancelled"]:
                logger.error(f"Batch {batch_id} {status}")
                error_count += 1
                
                # Update batch info
                batch_info["status"] = status
                batch_info["error_timestamp"] = time.time()
                
                with open(batch_info_path, "w") as f:
                    json.dump(batch_info, f)
        else:
            # Try to process all pending batches
            # Find all batch info files
            batch_info_files = [f for f in os.listdir(batch_dir) if f.startswith("batch_info_") and f.endswith(".json")]
            
            for batch_file in batch_info_files:
                batch_info_path = os.path.join(batch_dir, batch_file)
                
                with open(batch_info_path, "r") as f:
                    batch_info = json.load(f)
                
                # Skip already processed batches
                if batch_info.get("status") in ["processed", "error", "failed", "cancelled"]:
                    continue
                
                batch_id = batch_info["batch_id"]
                
                # Check batch status
                status = self._check_batch_status(batch_id)
                logger.info(f"Batch {batch_id} status: {status}")
                
                if status == "succeeded":
                    try:
                        # Fetch and process results
                        batch_results = self._fetch_batch_results(batch_id)
                        processed = self._process_batch_results(batch_results, batch_info["sample_identifiers"])
                        
                        # Update batch info
                        batch_info["status"] = "processed"
                        batch_info["processed_timestamp"] = time.time()
                        batch_info["processed_count"] = processed
                        
                        with open(batch_info_path, "w") as f:
                            json.dump(batch_info, f)
                        
                        processed_count += processed
                        logger.info(f"Processed {processed} samples from batch {batch_id}")
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_id}: {e}")
                        error_count += 1
                        
                        # Update batch info
                        batch_info["status"] = "error"
                        batch_info["error_timestamp"] = time.time()
                        batch_info["error_message"] = str(e)
                        
                        with open(batch_info_path, "w") as f:
                            json.dump(batch_info, f)
                elif status in ["failed", "cancelled"]:
                    logger.error(f"Batch {batch_id} {status}")
                    error_count += 1
                    
                    # Update batch info
                    batch_info["status"] = status
                    batch_info["error_timestamp"] = time.time()
                    
                    with open(batch_info_path, "w") as f:
                        json.dump(batch_info, f)
        
        logger.info(f"Batch result processing complete. Processed {processed_count} samples, errors {error_count}")
        
        return {
            "processed_count": processed_count,
            "error_count": error_count
        }

def main():
    """Main function to run the labeler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Label waypoints using ChatGPT")
    parser.add_argument("--data_dir", type=str, default="data/filtered_2k", help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="out/visualize_label", help="Directory to save visualizations")
    parser.add_argument("--n_waypoints", type=int, default=10, help="Number of waypoints")
    parser.add_argument("--max_labels", type=int, default=100, help="Maximum number of samples to label")
    parser.add_argument("--visualization_prob", type=float, default=0.1, help="Probability of visualizing a sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset before labeling")
    parser.add_argument("--manual", action="store_true", help="Use manual labeling mode")
    parser.add_argument("--use_batch", action="store_true", help="Use batch processing mode")
    parser.add_argument("--batch_size", type=int, default=3500, help="Batch size for batch processing")
    parser.add_argument("--fetch_batch", action="store_true", help="Fetch and process batch results")
    parser.add_argument("--batch_id", type=str, default=None, help="Specific batch ID to fetch")
    
    args = parser.parse_args()
    
    # Create the labeler
    labeler = WaypointLabeler(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_waypoints=args.n_waypoints,
        max_labels=args.max_labels,
        visualization_prob=args.visualization_prob,
        seed=args.seed,
        overwrite=args.overwrite,
        shuffle=args.shuffle,
        manual_mode=args.manual,
        use_batch=args.use_batch,
        batch_size=args.batch_size
    )
    
    if args.fetch_batch:
        # Fetch and process batch results
        stats = labeler.fetch_batch_results(args.batch_id)
        
        # Print statistics
        print("\nBatch Processing Statistics:")
        print(f"Processed samples: {stats['processed_count']}")
        print(f"Errors: {stats['error_count']}")
    else:
        # Run the labeling process
        stats = labeler.label_dataset()
        
        # Print final statistics
        print("\nLabeling Statistics:")
        print(f"Labeled samples: {stats['labeled_count']}")
        print(f"Skipped samples: {stats['skipped_count']}")
        print(f"Insufficient waypoints: {stats['insufficient_waypoints_count']}")
        print(f"Errors: {stats['error_count']}")
        if not args.manual:
            print(f"API calls: {stats['api_calls']}")
            print(f"Tokens used: {stats['tokens_used']}")

if __name__ == "__main__":
    main()
