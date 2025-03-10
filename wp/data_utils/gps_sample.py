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
            
            # Update counters
            location_counts[(city, country)] += 1
            country_counts[country] += 1
            city_counts[city] += 1
            ride_counts_by_country[country] += 1
            frame_counts_by_country[country] += frame_count
            
            logger.debug(f"Processed {ride_name}: {city}, {country}")
            
        except Exception as e:
            logger.error(f"Error processing GPS data for {ride_name}: {e}")
    
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

def sample_balanced_frames(data_dir, total_samples, output_dir="out/gps_stats"):
    """
    Sample frames such that each country has equal representation in the final sample.
    If a country has fewer frames than needed, the remaining quota will be distributed
    among other countries.
    
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
    
    # Group rides by country and collect all available frames
    rides_by_country = defaultdict(list)
    frames_by_country = defaultdict(list)
    
    for ride_dir in ride_dirs:
        meta_path = os.path.join(ride_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                country = meta.get('country', 'Unknown')
                if country in countries:
                    rides_by_country[country].append(ride_dir)
                    
                    # Collect frames for this ride
                    img_dir = os.path.join(ride_dir, "img")
                    if os.path.exists(img_dir):
                        img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
                        for img_file in img_files:
                            img_idx = int(os.path.basename(img_file).split('.')[0])
                            frames_by_country[country].append((ride_dir, img_idx))
    
    # First pass: sample what we can from each country
    sampled_frames = []
    remaining_quota = 0
    
    for country in countries:
        country_frames = frames_by_country[country]
        
        if len(country_frames) <= samples_per_country:
            # If we have fewer frames than needed, use all of them
            country_sampled = country_frames
            remaining_quota += (samples_per_country - len(country_frames))
            logger.warning(f"Country {country} has only {len(country_frames)} frames, " 
                          f"less than the {samples_per_country} requested")
        else:
            # Otherwise, randomly sample the required number
            country_sampled = random.sample(country_frames, samples_per_country)
        
        sampled_frames.extend(country_sampled)
        
        # Remove sampled frames from available frames
        for frame in country_sampled:
            if frame in frames_by_country[country]:
                frames_by_country[country].remove(frame)
    
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
                extra_to_take = min(extra_to_take, len(frames_by_country[country]))
                
                if extra_to_take > 0:
                    extra_sampled = random.sample(frames_by_country[country], extra_to_take)
                    sampled_frames.extend(extra_sampled)
                    logger.info(f"Added {len(extra_sampled)} extra frames from {country}")
    
    # Shuffle the final list
    random.shuffle(sampled_frames)
    
    # Save the sampled frames list
    sampled_frames_path = os.path.join(output_dir, f"balanced_frames_{total_samples}.pkl")
    with open(sampled_frames_path, 'wb') as f:
        pickle.dump(sampled_frames, f)
    
    logger.info(f"Sampled a total of {len(sampled_frames)} frames")
    logger.info(f"Saved balanced frame list to {sampled_frames_path}")
    
    return sampled_frames

def collect_sampled_frames(data_dir, output_dir, total_samples, divide=None):
    """
    Collect all sampled frames and copy them to a specified output directory.
    Optionally divide the frames into N sub-folders with equal distribution.
    
    Args:
        data_dir (str): Path to the data directory
        output_dir (str): Directory to save collected frames
        total_samples (int): Total number of frames to sample
        divide (int, optional): Number of sub-folders to divide frames into
    
    Returns:
        list: List of paths to the copied frames
    """
    import shutil
    
    logger.info(f"Collecting sampled frames to {output_dir}...")
    
    # Get sampled frames
    stats_dir = os.path.join(os.path.dirname(output_dir), "gps_stats")
    sampled_frames = sample_balanced_frames(data_dir, total_samples, output_dir=stats_dir)
    
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
    for i, (ride_dir, img_idx) in enumerate(sampled_frames):
        # Source image path
        src_path = os.path.join(ride_dir, "img", f"{img_idx}.jpg")
        
        # Determine destination path
        if divide and divide > 0:
            # Determine which sub-folder this frame goes into
            folder_idx = i // (frames_per_folder + (1 if i < remainder else 0))
            folder_idx = min(folder_idx, divide - 1)  # Ensure we don't exceed the number of folders
            dest_folder = os.path.join(output_dir, f"subset_{folder_idx+1}")
        else:
            dest_folder = output_dir
        
        # Create a unique filename using ride directory and frame index
        ride_name = os.path.basename(ride_dir)
        dest_filename = f"{ride_name}_{img_idx}.jpg"
        dest_path = os.path.join(dest_folder, dest_filename)
        
        # Copy the file
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied_frames.append(dest_path)
        else:
            logger.warning(f"Source file not found: {src_path}")
    
    logger.info(f"Collected {len(copied_frames)} frames to {output_dir}")
    return copied_frames

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
    country_df = country_df.sort_values('Frames', ascending=False)
    
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
    logger.info(f"Top 5 countries by frame count:")
    for _, row in country_df.head(5).iterrows():
        logger.info(f"  {row['Country']}: {row['Frames']} frames ({row['Frames_Percentage']:.2f}%)")
    
    return country_df

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
    
    args = parser.parse_args()
    
    # Process GPS data if requested
    if args.process:
        stats = process_gps_data(args.data_dir, output_dir=args.output_dir)
    
    # Generate report if requested
    if args.report:
        report = generate_location_report(args.data_dir, output_dir=args.output_dir)
    
    # Sample frames if requested
    if args.sample > 0:
        if args.collect:
            # Collect sampled frames
            collect_sampled_frames(args.data_dir, args.collect_dir, args.sample, divide=args.divide)
        else:
            # Just sample frames without collecting
            sampled_frames = sample_balanced_frames(args.data_dir, args.sample, output_dir=args.output_dir)




