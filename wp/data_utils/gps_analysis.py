import pickle
import numpy as np
import os
import glob
from tqdm import tqdm
import logging
import random
from collections import Counter, defaultdict
import pandas as pd
import time
from geopy.geocoders import Nominatim
import pytz
from datetime import datetime
from timezonefinder import TimezoneFinder
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_gps_data(file_path):
    """Load GPS data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        latitude = data['latitude']
        longitude = data['longitude']
        timestamp = data['timestamp']
    
    # return the average of the latitude and longitude
    return np.mean(latitude), np.mean(longitude)

def get_location_info(latitude, longitude, max_retries=3, delay=1):
    """Get location information (city, country) from coordinates using Nominatim."""
    geolocator = Nominatim(user_agent="waypoint_predictor")
    
    for attempt in range(max_retries):
        try:
            location = geolocator.reverse(f"{latitude}, {longitude}", language="en")
            if location and location.raw.get('address'):
                address = location.raw['address']
                country = address.get('country', 'Unknown')
                city = address.get('city', address.get('town', address.get('village', 'Unknown')))
                return city, country
            time.sleep(delay)  # Be nice to the API
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(delay * (attempt + 1))  # Exponential backoff
    
    return "Unknown", "Unknown"

def process_gps_data(data_dir, output_dir="out/gps_stats"):
    """
    Process GPS data for all rides and save location metadata.
    
    Args:
        data_dir (str): Path to the data directory containing ride folders
    
    Returns:
        dict: Statistics about the dataset
    """
    logger.info("Processing GPS data for all rides...")
    
    # Find all ride directories
    ride_dirs = sorted(glob.glob(os.path.join(data_dir, "ride_*")))
    logger.info(f"Found {len(ride_dirs)} ride directories")
    
    # Initialize counters
    location_counts = Counter()
    country_counts = Counter()
    city_counts = Counter()
    ride_counts_by_country = defaultdict(int)
    frame_counts_by_country = defaultdict(int)
    
    # Process each ride
    for ride_dir in tqdm(ride_dirs):
        ride_name = os.path.basename(ride_dir)
        gps_path = os.path.join(ride_dir, "gps_raw.pkl")
        meta_path = os.path.join(ride_dir, "meta.pkl")
        
        # if meta.pkl exists, skip
        if not os.path.exists(meta_path):
        
            # Skip if GPS data doesn't exist
            if not os.path.exists(gps_path):
                logger.warning(f"No GPS data found for {ride_name}")
                continue
            
            try:
                # Load GPS data and get average coordinates
                avg_lat, avg_lon = load_gps_data(gps_path)
                
                # Get location information
                city, country = get_location_info(avg_lat, avg_lon)
                
                # Count frames in this ride
                img_dir = os.path.join(ride_dir, "img")
                if os.path.exists(img_dir):
                    frame_count = len(glob.glob(os.path.join(img_dir, "*.jpg")))
                else:
                    frame_count = 0
                
                # Save metadata
                metadata = {
                    'latitude': avg_lat,
                    'longitude': avg_lon,
                    'city': city,
                    'country': country,
                    'frame_count': frame_count
                }
                
                with open(meta_path, 'wb') as f:
                    pickle.dump(metadata, f) 
            except Exception as e:
                logger.error(f"Error processing GPS data for {ride_name}: {e}")
            
        else:
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
                city = metadata['city']
                country = metadata['country']
                frame_count = metadata['frame_count']
            
            # Update counters
            location_counts[(city, country)] += 1
            country_counts[country] += 1
            city_counts[city] += 1
            ride_counts_by_country[country] += 1
            frame_counts_by_country[country] += frame_count
            
            logger.debug(f"Processed {ride_name}: {city}, {country}")
            
       
    
    # Compile statistics
    stats = {
        'total_rides': len(ride_dirs),
        'rides_with_location': sum(location_counts.values()),
        'unique_countries': len(country_counts),
        'unique_cities': len(city_counts),
        'country_distribution': dict(country_counts),
        'city_distribution': dict(city_counts),
        'ride_counts_by_country': dict(ride_counts_by_country),
        'frame_counts_by_country': dict(frame_counts_by_country)
    }
    
    # Save overall statistics
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "location_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    # Print summary
    logger.info(f"GPS data processing complete.")
    logger.info(f"Total rides: {stats['total_rides']}")
    logger.info(f"Rides with location data: {stats['rides_with_location']}")
    logger.info(f"Unique countries: {stats['unique_countries']}")
    logger.info(f"Unique cities: {stats['unique_cities']}")
    logger.info(f"Country distribution: {stats['country_distribution']}")
    logger.info(f"Frame counts by country: {stats['frame_counts_by_country']}")
    
    return stats

def sample_frames(data_dir, total_samples, output_dir="out/gps_stats"):
    """
    Sample frames with diverse locations and orientations in a single pass.
    Uses both GPS location and compass heading to ensure diversity.
    
    Args:
        data_dir (str): Path to the data directory
        total_samples (int): Total number of frames to sample
        output_dir (str): Directory to save output statistics
    
    Returns:
        list: List of sampled frame paths (ride_dir, img_idx, coords, heading)
    """
    logger.info(f"Sampling up to {total_samples} frames with diverse locations and orientations...")
    
    # Find all ride directories
    ride_dirs = sorted(glob.glob(os.path.join(data_dir, "ride_*")))
    logger.info(f"Found {len(ride_dirs)} ride directories")
    
    # Constants for Earth radius and conversion
    EARTH_RADIUS_METERS = 6371000  # Earth's radius in meters
    METERS_PER_DEGREE_LAT = EARTH_RADIUS_METERS * (np.pi / 180)  # ~111,000 meters per degree
    TARGET_GRID_SIZE_METERS = 3  # Target grid size in meters
    HEADING_GRID_SIZE_DEGREES = 20  # Divide 360 degrees into 8 bins
    MAX_FRAMES_PER_CELL = 5
    
    # Use a spatial-heading grid for efficient proximity checking
    location_heading_grid = defaultdict(int)  # Will return 0 for any new cell_key
    
    # Collect all available frames with GPS and compass data
    all_frames = []
    
    for ride_dir in tqdm(ride_dirs, desc="Collecting frames with GPS and compass data"):
        gps_path = os.path.join(ride_dir, "gps_raw.pkl")
        compass_path = os.path.join(ride_dir, "compass_calibrated.pkl")
        
        # Skip if GPS data doesn't exist
        if not os.path.exists(gps_path):
            continue
            
        # Load GPS data
        with open(gps_path, 'rb') as gps_f:
            gps_data = pickle.load(gps_f)
            latitudes = gps_data['latitude']
            longitudes = gps_data['longitude']
            gps_timestamps = gps_data['timestamp']
        
        # Load compass data if available
        compass_headings = None
        compass_timestamps = None
        if os.path.exists(compass_path):
            with open(compass_path, 'rb') as compass_f:
                compass_data = pickle.load(compass_f)
                # First column is heading angle in degrees, second is timestamp
                compass_headings = compass_data[:, 0]
                compass_timestamps = compass_data[:, 1]
        
        # Collect frames for this ride
        img_dir = os.path.join(ride_dir, "img")
        if os.path.exists(img_dir):
            img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
            for img_file in img_files:
                img_idx = int(os.path.basename(img_file).split('.')[0])
                
                # Find frame timestamp
                traj_path = os.path.join(ride_dir, "traj_data.pkl")
                if os.path.exists(traj_path):
                    with open(traj_path, 'rb') as traj_f:
                        traj_data = pickle.load(traj_f)
                        if 'timestamps' in traj_data and len(traj_data['timestamps']) > img_idx:
                            frame_timestamp = traj_data['timestamps'][img_idx]
                            
                            # Find closest GPS timestamp
                            closest_gps_idx = np.argmin(np.abs(np.array(gps_timestamps) - frame_timestamp))
                            lat = latitudes[closest_gps_idx]
                            lon = longitudes[closest_gps_idx]
                            
                            # Find closest compass heading if available
                            heading = None
                            if compass_headings is not None and compass_timestamps is not None:
                                closest_compass_idx = np.argmin(np.abs(np.array(compass_timestamps) - frame_timestamp))
                                heading = compass_headings[closest_compass_idx]
                            
                            all_frames.append((ride_dir, img_idx, (lat, lon), heading))
    
    logger.info(f"Collected {len(all_frames)} frames with GPS data")
    
    # Shuffle all frames to ensure random selection
    random.shuffle(all_frames)
    
    # Sample frames while respecting the GPS proximity and heading constraint
    sampled_frames = []
    
    for frame in tqdm(all_frames, desc="Sampling diverse frames"):
        ride_dir, img_idx, (lat, lon), heading = frame
        
        # Calculate grid size for longitude at this latitude
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * np.cos(np.radians(lat))
        
        # Calculate grid cell size in degrees
        lat_grid_size = TARGET_GRID_SIZE_METERS / METERS_PER_DEGREE_LAT
        lon_grid_size = TARGET_GRID_SIZE_METERS / meters_per_degree_lon if meters_per_degree_lon > 0 else lat_grid_size
        
        # Convert to grid cell coordinates
        grid_x = int(lon / lon_grid_size)
        grid_y = int(lat / lat_grid_size)
        
        # Add heading to grid if available
        if heading is not None:
            # Normalize heading to 0-360 range
            heading = heading % 360
            # Convert to grid cell
            grid_heading = int(heading / HEADING_GRID_SIZE_DEGREES)
            grid_key = (grid_x, grid_y, grid_heading)
        else:
            # If no heading data, just use location
            grid_key = (grid_x, grid_y)
        
        # Check if this cell already has MAX_FRAMES_PER_CELL or more frames
        if location_heading_grid[grid_key] >= MAX_FRAMES_PER_CELL:
            continue
  
        sampled_frames.append((ride_dir, img_idx, (lat, lon), heading))
        location_heading_grid[grid_key] += 1
            
        # Break if we've sampled enough frames
        if len(sampled_frames) >= total_samples:
            break
    
    logger.info(f"Sampled a total of {len(sampled_frames)} frames")
    
    # Save the sampled frames list
    os.makedirs(output_dir, exist_ok=True)
    sampled_frames_path = os.path.join(output_dir, f"sampled_frames_{total_samples}.pkl")
    with open(sampled_frames_path, 'wb') as f:
        pickle.dump(sampled_frames, f)
    
    logger.info(f"Saved frame list to {sampled_frames_path}")
    
    return sampled_frames

def sample_balanced_frames(data_dir, total_samples, output_dir="out/gps_stats"):
    """
    Sample frames such that each country has equal representation in the final sample.
    If a country has fewer frames than needed, the remaining quota will be distributed
    among other countries. Also ensures that within 10 meters of any GPS location, 
    we only sample a maximum of 5 frames.
    
    Args:
        data_dir (str): Path to the data directory
        total_samples (int): Total number of frames to sample
        output_dir (str): Directory to save output statistics
    
    Returns:
        list: List of sampled frame paths (ride_dir, img_idx)
    """
    logger.info(f"Sampling {total_samples} frames with balanced country distribution...")
    
    # Check if location stats exist, if not, process GPS data
    stats_path = os.path.join(output_dir, "location_stats.pkl")
    if not os.path.exists(stats_path):
        process_gps_data(data_dir)
    
    # Load location stats
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Get countries with frame data
    countries = [country for country, count in stats['frame_counts_by_country'].items() 
                if count > 0 and country != "Unknown"]
    
    if not countries:
        logger.error("No countries with frame data found!")
        return []
    
    # Calculate initial samples per country
    samples_per_country = total_samples // len(countries)
    logger.info(f"Target sampling {samples_per_country} frames from each of {len(countries)} countries")
    
    # Find all ride directories
    ride_dirs = sorted(glob.glob(os.path.join(data_dir, "ride_*")))
    
    # Group rides by country and collect all available frames with GPS data
    rides_by_country = defaultdict(list)
    frames_by_country = defaultdict(list)
    
    for ride_dir in tqdm(ride_dirs, desc="Collecting frames with GPS data"):
        meta_path = os.path.join(ride_dir, "meta.pkl")
        gps_path = os.path.join(ride_dir, "gps_raw.pkl")
        
        if not os.path.exists(meta_path) or not os.path.exists(gps_path):
            continue
            
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            country = meta.get('country', 'Unknown')
            if country in countries:
                rides_by_country[country].append(ride_dir)
                
                # Load GPS data
                with open(gps_path, 'rb') as gps_f:
                    gps_data = pickle.load(gps_f)
                    latitudes = gps_data['latitude']
                    longitudes = gps_data['longitude']
                    timestamps = gps_data['timestamp']
                
                # Collect frames for this ride
                img_dir = os.path.join(ride_dir, "img")
                if os.path.exists(img_dir):
                    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
                    for img_file in img_files:
                        img_idx = int(os.path.basename(img_file).split('.')[0])
                        
                        # Find closest timestamp in GPS data
                        traj_path = os.path.join(ride_dir, "traj_data.pkl")
                        if os.path.exists(traj_path):
                            with open(traj_path, 'rb') as traj_f:
                                traj_data = pickle.load(traj_f)
                                if 'timestamps' in traj_data and len(traj_data['timestamps']) > img_idx:
                                    frame_timestamp = traj_data['timestamps'][img_idx]
                                    
                                    # Find closest GPS timestamp
                                    closest_idx = np.argmin(np.abs(np.array(timestamps) - frame_timestamp))
                                    lat = latitudes[closest_idx]
                                    lon = longitudes[closest_idx]
                                    
                                    frames_by_country[country].append((ride_dir, img_idx, (lat, lon)))
    
    # First pass: sample what we can from each country
    sampled_frames = []
    remaining_quota = 0
    
    # Use a spatial grid for efficient proximity checking
    # Each grid cell is approximately 10x10 meters
    # We'll adjust longitude grid size based on latitude to account for Earth's curvature
    location_grid = defaultdict(int)  # Will return 0 for any new cell_key
    
    # Constants for Earth radius and conversion
    EARTH_RADIUS_METERS = 6371000  # Earth's radius in meters
    METERS_PER_DEGREE_LAT = EARTH_RADIUS_METERS * (np.pi / 180)  # ~111,000 meters per degree
    TARGET_GRID_SIZE_METERS = 3  # Target grid size in meters
    MAX_FRAMES_PER_CELL = 5
    
    logger.info("Sampling frames with balanced distribution...")
    for country in countries:
        country_frames = frames_by_country[country]
        
        # Shuffle frames to ensure random selection
        random.shuffle(country_frames)
        
        # Sample frames while respecting the GPS proximity constraint
        country_sampled = []
        for frame in country_frames:
            ride_dir, img_idx, (lat, lon) = frame
            
            # Calculate grid size for longitude at this latitude
            # At the equator, 1 degree longitude = 111km, but this decreases with latitude
            meters_per_degree_lon = METERS_PER_DEGREE_LAT * np.cos(np.radians(lat))
            
            # Calculate grid cell size in degrees
            lat_grid_size = TARGET_GRID_SIZE_METERS / METERS_PER_DEGREE_LAT
            lon_grid_size = TARGET_GRID_SIZE_METERS / meters_per_degree_lon if meters_per_degree_lon > 0 else lat_grid_size
            
            # Convert to grid cell coordinates
            grid_x = int(lon / lon_grid_size)
            grid_y = int(lat / lat_grid_size)
            
            # Check if this cell already has 5 or more frames
            if location_grid[(grid_x, grid_y)] >= MAX_FRAMES_PER_CELL:
                continue
      
            country_sampled.append((ride_dir, img_idx))
            location_grid[(grid_x, grid_y)] += 1
                
            # Break if we've sampled enough frames
            if len(country_sampled) >= samples_per_country:
                break
        
        # If we couldn't sample enough frames, add to remaining quota
        if len(country_sampled) < samples_per_country:
            remaining_quota += (samples_per_country - len(country_sampled))
            logger.warning(f"Country {country} has only {len(country_sampled)} valid frames, " 
                          f"less than the {samples_per_country} requested")
        
        # Add sampled frames to the final list
        sampled_frames.extend(country_sampled)
        
        # Remove sampled frames from available frames
        for ride_dir, img_idx in country_sampled:
            frames_to_remove = [(r, i, coords) for r, i, coords in frames_by_country[country] if r == ride_dir and i == img_idx]
            for f in frames_to_remove:
                if f in frames_by_country[country]:
                    frames_by_country[country].remove(f)
    
    # Second pass: distribute remaining quota among countries with extra frames
    if remaining_quota > 0:
        logger.info(f"Redistributing remaining quota of {remaining_quota} frames")
        
        # Find countries with remaining frames
        countries_with_extra = [c for c in countries if len(frames_by_country[c]) > 0]
        
        if countries_with_extra:
            # Distribute remaining quota as evenly as possible
            extra_per_country = remaining_quota // len(countries_with_extra)
            remainder = remaining_quota % len(countries_with_extra)
            
            for i, country in enumerate(countries_with_extra):
                # Add remainder to first few countries if needed
                extra_to_take = extra_per_country + (1 if i < remainder else 0)
                
                # Sample additional frames while respecting GPS proximity constraint
                extra_sampled = []
                for frame in frames_by_country[country]:
                    ride_dir, img_idx, (lat, lon) = frame
                    
                    # Calculate grid size for longitude at this latitude
                    meters_per_degree_lon = METERS_PER_DEGREE_LAT * np.cos(np.radians(lat))
                    
                    # Calculate grid cell size in degrees
                    lat_grid_size = TARGET_GRID_SIZE_METERS / METERS_PER_DEGREE_LAT
                    lon_grid_size = TARGET_GRID_SIZE_METERS / meters_per_degree_lon if meters_per_degree_lon > 0 else lat_grid_size
                    
                    # Convert to grid cell coordinates
                    grid_x = int(lon / lon_grid_size)
                    grid_y = int(lat / lat_grid_size)
                    
                    # Check if this cell already has 5 or more frames
                    if location_grid[(grid_x, grid_y)] >= MAX_FRAMES_PER_CELL:
                        continue

                    extra_sampled.append((ride_dir, img_idx))
                    location_grid[(grid_x, grid_y)] += 1
                        
                    # Break if we've sampled enough extra frames
                    if len(extra_sampled) >= extra_to_take:
                        break
                
                sampled_frames.extend(extra_sampled)
                logger.info(f"Added {len(extra_sampled)} extra frames from {country}")
    
    # Shuffle the final list
    random.shuffle(sampled_frames)
    
    # Save the sampled frames list with GPS coordinates
    sampled_frames_with_gps = []
    for ride_dir, img_idx in sampled_frames:
        # Find the GPS coordinates for this frame
        for country in countries:
            for frame in frames_by_country[country]:
                if frame[0] == ride_dir and frame[1] == img_idx:
                    sampled_frames_with_gps.append((ride_dir, img_idx, frame[2]))
                    break
    
    sampled_frames_path = os.path.join(output_dir, f"balanced_frames_{total_samples}.pkl")
    with open(sampled_frames_path, 'wb') as f:
        pickle.dump(sampled_frames_with_gps, f)
    
    logger.info(f"Sampled a total of {len(sampled_frames)} frames")
    logger.info(f"Saved balanced frame list to {sampled_frames_path}")
    
    return sampled_frames


def collect_sampled_frames(data_dir, output_dir, total_samples, divide=None, future_waypoints=10, balanced=True):
    """
    Collect all sampled frames and copy them to a specified output directory.
    Optionally divide the frames into N sub-folders with equal distribution.
    Also saves the corresponding future waypoints for each frame in the local frame.
    
    Args:
        data_dir (str): Path to the data directory
        output_dir (str): Directory to save collected frames
        total_samples (int): Total number of frames to sample
        divide (int, optional): Number of sub-folders to divide frames into
    
    Returns:
        list: List of paths to the copied frames
    """
    
    logger.info(f"Collecting sampled frames to {output_dir}...")
    
    # Get sampled frames
    stats_dir = os.path.join(os.path.dirname(output_dir), "gps_stats")
    if balanced:
        sampled_frames = sample_balanced_frames(data_dir, total_samples, output_dir=stats_dir)
    else:
        logger.info("Not using balanced sampling")
        sampled_frames = sample_frames(data_dir, total_samples, output_dir=stats_dir)
    
    if not sampled_frames:
        logger.error("No frames were sampled!")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine sub-folder structure
    if divide and divide > 0:
        frames_per_folder = len(sampled_frames) // divide
        remainder = len(sampled_frames) % divide
        logger.info(f"Dividing {len(sampled_frames)} frames into {divide} folders " 
                   f"with approximately {frames_per_folder} frames each")
        
        # Create sub-folders
        for i in range(divide):
            os.makedirs(os.path.join(output_dir, f"subset_{i+1}"), exist_ok=True)
    
    # Copy frames to output directory
    copied_frames = []
    for i, frame_data in enumerate(sampled_frames):
        # Handle different return formats from sample_frames and sample_balanced_frames
        if balanced:
            if len(frame_data) == 3:  # (ride_dir, img_idx, coords)
                ride_dir, img_idx, coords = frame_data
                lat, lon = coords
            else:  # Just (ride_dir, img_idx)
                ride_dir, img_idx = frame_data
                lat, lon = None, None
        else:
            if len(frame_data) == 4:  # (ride_dir, img_idx, coords, heading)
                ride_dir, img_idx, coords, heading = frame_data
                lat, lon = coords
            else:  # Unexpected format
                logger.warning(f"Unexpected frame data format: {frame_data}")
                continue
        
        # Source image path
        src_path = os.path.join(ride_dir, "img", f"{img_idx}.jpg")
        
        if future_waypoints > 0:
            # Load trajectory data to get waypoints
            traj_path = os.path.join(ride_dir, "traj_data.pkl")
            if not os.path.exists(traj_path):
                logger.warning(f"No trajectory data found for {ride_dir}")
                continue
                
            with open(traj_path, "rb") as f:
                traj_data = pickle.load(f)
                
            # Get positions data
            pos = traj_data["pos"]
            
            # Check if there are enough future waypoints
            if img_idx + future_waypoints >= len(pos):
                logger.warning(f"Not enough future waypoints for {ride_dir}, frame {img_idx}")
                continue
                
            # Extract the next n_waypoints
            future_waypoints_data = pos[img_idx+1:img_idx+future_waypoints+1].copy()
            
            # Get current position and orientation for local frame transformation
            current_pos = pos[img_idx].copy()
            current_yaw = traj_data["yaw"][img_idx]
            
            # Transform waypoints to local frame
            local_waypoints = _transform_to_local_frame(future_waypoints_data, current_pos, current_yaw)
        
        # Determine destination path
        if divide and divide > 0:
            # Determine which sub-folder this frame goes into
            folder_idx = i // (frames_per_folder + (1 if i < remainder else 0))
            folder_idx = min(folder_idx, divide - 1)  # Ensure we don't exceed the number of folders
            dest_folder = os.path.join(output_dir, f"subset_{folder_idx+1}")
        else:
            dest_folder = output_dir
        
        os.makedirs(dest_folder, exist_ok=True)
        
        # Create a unique filename using ride directory and frame index
        ride_name = os.path.basename(ride_dir)
        dest_filename = f"{ride_name}_{img_idx}.jpg"
        dest_path = os.path.join(dest_folder, dest_filename)
        
        # Copy the image file
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            
            copied_frames.append(dest_path)

            if future_waypoints > 0:
                # Create waypoints filename
                waypoints_filename = f"{ride_name}_{img_idx}_waypoints.npy"
                waypoints_path = os.path.join(dest_folder, waypoints_filename)
                # Save the waypoints
                np.save(waypoints_path, local_waypoints)
        else:
            logger.warning(f"Source file not found: {src_path}")
    
    logger.info(f"Collected {len(copied_frames)} frames to {output_dir}")
    return copied_frames

def _transform_to_local_frame(waypoints, current_pos, current_yaw):
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

def generate_location_report(data_dir, output_dir="out/gps_stats", output_file="location_report.csv"):
    """
    Generate a detailed report of location statistics.
    
    Args:
        data_dir (str): Path to the data directory
        output_file (str, optional): Path to save the report (CSV format)
    
    Returns:
        pd.DataFrame: DataFrame containing the report
    """
    # Check if location stats exist, if not, process GPS data
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "location_stats.pkl")
    if not os.path.exists(stats_path):
        process_gps_data(data_dir)
    
    # Load location stats
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # Create DataFrame for country statistics
    country_data = []
    for country, ride_count in stats['ride_counts_by_country'].items():
        frame_count = stats['frame_counts_by_country'].get(country, 0)
        country_data.append({
            'Country': country,
            'Rides': ride_count,
            'Frames': frame_count,
            'Frames_Percentage': frame_count / sum(stats['frame_counts_by_country'].values()) * 100 if sum(stats['frame_counts_by_country'].values()) > 0 else 0
        })
    
    country_df = pd.DataFrame(country_data)
    
    # Only sort if the DataFrame is not empty and contains the 'Frames' column
    if not country_df.empty and 'Frames' in country_df.columns:
        country_df = country_df.sort_values('Frames', ascending=False)
    else:
        logger.warning("Cannot sort DataFrame: either empty or missing 'Frames' column")
    
    # Save to CSV if output file is specified
    if output_file:
        output_file = os.path.join(output_dir, output_file)
        country_df.to_csv(output_file, index=False)
        logger.info(f"Location report saved to {output_file}")
    
    # Print summary
    logger.info("\nLocation Statistics Summary:")
    logger.info(f"Total rides: {stats['total_rides']}")
    logger.info(f"Total frames: {sum(stats['frame_counts_by_country'].values())}")
    logger.info(f"Unique countries: {stats['unique_countries']}")
    
    # Only print top countries if there are any
    if not country_df.empty and 'Frames' in country_df.columns:
        logger.info(f"Top 5 countries by frame count:")
        for _, row in country_df.head(5).iterrows():
            logger.info(f"  {row['Country']}: {row['Frames']} frames ({row['Frames_Percentage']:.2f}%)")
    else:
        logger.info("No country frame data available to display")
    
    return country_df

def filter_night_rides(data_dir, output_dir="out/gps_stats"):
    """
    Filter out rides that were recorded during nighttime (7 PM to 7 AM local time).
    
    Args:
        data_dir (str): Path to the data directory containing ride folders
        output_dir (str): Directory to save output statistics
    
    Returns:
        dict: Statistics about filtered rides
    """
    logger.info("Filtering out night-time rides (7 PM to 7 AM local time)...")
    
    # Find all ride directories
    ride_dirs = sorted(glob.glob(os.path.join(data_dir, "ride_*")))
    logger.info(f"Found {len(ride_dirs)} ride directories")
    
    # Initialize timezone finder
    tf = TimezoneFinder()
    
    # Initialize statistics
    stats = {
        'total_rides': len(ride_dirs),
        'night_rides': 0,
        'day_rides': 0,
        'no_gps_data': 0,
        'no_timestamp_data': 0,
        'processing_errors': 0,
        'countries': {}  # Changed from defaultdict with lambda to regular dict
    }
    
    # Process each ride
    for ride_dir in tqdm(ride_dirs):
        ride_name = os.path.basename(ride_dir)
        gps_path = os.path.join(ride_dir, "gps_raw.pkl")
        meta_path = os.path.join(ride_dir, "meta.pkl")
        
        try:
            # Check if GPS data exists
            if not os.path.exists(gps_path):
                logger.warning(f"No GPS data found for {ride_name}")
                stats['no_gps_data'] += 1
                continue
            
            # Load or create metadata
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)
            else:
                # Load GPS data and get average coordinates
                avg_lat, avg_lon = load_gps_data(gps_path)
                
                # Get location information
                city, country = get_location_info(avg_lat, avg_lon)
                
                # Count frames in this ride
                img_dir = os.path.join(ride_dir, "img")
                if os.path.exists(img_dir):
                    frame_count = len(glob.glob(os.path.join(img_dir, "*.jpg")))
                else:
                    frame_count = 0
                
                metadata = {
                    'latitude': avg_lat,
                    'longitude': avg_lon,
                    'city': city,
                    'country': country,
                    'frame_count': frame_count
                }
            
            # Get timestamp data
            traj_path = os.path.join(ride_dir, "traj_data.pkl")
            if not os.path.exists(traj_path):
                logger.warning(f"No trajectory data found for {ride_name}")
                stats['no_timestamp_data'] += 1
                continue
                
            with open(traj_path, 'rb') as f:
                traj_data = pickle.load(f)
                
            if 'timestamps' not in traj_data or len(traj_data['timestamps']) == 0:
                logger.warning(f"No timestamp data found in trajectory for {ride_name}")
                stats['no_timestamp_data'] += 1
                continue
            
            # Get the first timestamp (Unix timestamp in seconds)
            unix_timestamp = traj_data['timestamps'][0]
            
            # Get timezone from coordinates
            lat, lon = metadata.get('latitude'), metadata.get('longitude')
            timezone_str = tf.timezone_at(lat=lat, lng=lon)
            
            if not timezone_str:
                logger.warning(f"Could not determine timezone for {ride_name} at {lat}, {lon}")
                timezone_str = 'UTC'  # Default to UTC if timezone can't be determined
            
            # Convert Unix timestamp to local time
            timezone = pytz.timezone(timezone_str)
            utc_time = datetime.utcfromtimestamp(unix_timestamp)
            utc_time = pytz.utc.localize(utc_time)
            local_time = utc_time.astimezone(timezone)
            
            # Add local time info to metadata
            metadata['local_time'] = local_time.strftime('%Y-%m-%d %H:%M:%S')
            metadata['timezone'] = timezone_str
            metadata['hour_of_day'] = local_time.hour
            metadata['is_night'] = local_time.hour >= 19 or local_time.hour < 7
            
            # Save updated metadata
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Check if it's nighttime (7 PM to 7 AM)
            is_night = metadata['is_night']
            country = metadata.get('country', 'Unknown')
            
            # Initialize country stats if not already present
            if country not in stats['countries']:
                stats['countries'][country] = {'day': 0, 'night': 0}
            
            if is_night:
                stats['night_rides'] += 1
                stats['countries'][country]['night'] += 1
                logger.debug(f"Night ride detected: {ride_name} at {metadata['local_time']} in {country}")
                
                # Option to remove or move the night ride
                # Uncomment one of these options if you want to actually remove the rides
                
                # Option 1: Delete the ride directory
                shutil.rmtree(ride_dir)
                
                # # Option 2: Move to a "night_rides" directory
                # night_dir = os.path.join(os.path.dirname(data_dir), "night_rides")
                # os.makedirs(night_dir, exist_ok=True)
                # shutil.move(ride_dir, os.path.join(night_dir, ride_name))
            else:
                stats['day_rides'] += 1
                stats['countries'][country]['day'] += 1
                logger.debug(f"Day ride detected: {ride_name} at {metadata['local_time']} in {country}")
            
        except Exception as e:
            logger.error(f"Error processing ride {ride_name}: {e}")
            stats['processing_errors'] += 1
    
    # Save statistics
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "day_night_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    # Print summary
    logger.info(f"Day/Night filtering complete.")
    logger.info(f"Total rides: {stats['total_rides']}")
    logger.info(f"Day rides: {stats['day_rides']}")
    logger.info(f"Night rides: {stats['night_rides']}")
    logger.info(f"Rides without GPS data: {stats['no_gps_data']}")
    logger.info(f"Rides without timestamp data: {stats['no_timestamp_data']}")
    logger.info(f"Rides with processing errors: {stats['processing_errors']}")
    
    # Print country breakdown
    logger.info("Country breakdown (day/night):")
    for country, counts in sorted(stats['countries'].items(), key=lambda x: x[1]['day'] + x[1]['night'], reverse=True):
        total = counts['day'] + counts['night']
        if total > 0:
            night_percent = (counts['night'] / total) * 100
            logger.info(f"  {country}: {counts['day']} day, {counts['night']} night ({night_percent:.1f}% night)")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPS data processing and frame sampling")
    parser.add_argument("--data_dir", default="data/filtered_2k", help="Path to data directory")
    parser.add_argument("--output_dir", default="out/gps_stats", help="Path to output directory")
    parser.add_argument("--process", action="store_true", help="Process GPS data and generate statistics")
    parser.add_argument("--report", action="store_true", help="Generate location report")
    parser.add_argument("--sample", type=int, default=0, help="Number of frames to sample with balanced distribution")
    parser.add_argument("--collect", action="store_true", help="Collect sampled frames to output directory")
    parser.add_argument("--collect_dir", default="out/sampled_frames", help="Directory to save collected frames")
    parser.add_argument("--divide", type=int, help="Divide collected frames into N sub-folders")
    parser.add_argument("--future_waypoints", type=int, default=-1, help="Number of future waypoints to collect")
    parser.add_argument("--filter_night", action="store_true", help="Filter out night-time rides (7 PM to 7 AM)")
    parser.add_argument("--balanced", action="store_true", help="Use balanced sampling")
    args = parser.parse_args()
    
    # Process GPS data if requested
    if args.process:
        stats = process_gps_data(args.data_dir, output_dir=args.output_dir)
    
    # Generate report if requested
    if args.report:
        report = generate_location_report(args.data_dir, output_dir=args.output_dir)
    
    # Filter night rides if requested
    if args.filter_night:
        night_stats = filter_night_rides(args.data_dir, output_dir=args.output_dir)
    
    # Sample frames if requested
    if args.sample > 0:
        if args.collect:
            # Collect sampled frames
            collect_sampled_frames(
                args.data_dir, 
                args.collect_dir, 
                args.sample, 
                divide=args.divide, 
                future_waypoints=args.future_waypoints, 
                balanced=args.balanced)
        else:
            # Just sample frames without collecting
            sampled_frames = sample_balanced_frames(args.data_dir, args.sample, output_dir=args.output_dir)




