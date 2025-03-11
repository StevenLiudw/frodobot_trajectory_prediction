import csv
import re
import numpy as np
import math
from scipy.optimize import least_squares
from scipy.linalg import eigh
# from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import utm
import scipy
import geomag

USE_NED = True
def visualize_headings(
    gps_data, 
    heading_data, 
    heading_data_key_name="heading",
    output_path=None,
    title="Compass Headings", 
    metrics=None, 
    max_arrows=150
    ):
    """
    Visualize compass headings as arrows at GPS locations
    
    Args:
        gps_data: List of dictionaries with GPS data
        heading_data: List of dictionaries with heading data
        heading_data_key_name: Key name for heading in heading_data dictionaries
        output_path: Path to save the output image
        title: Title for the plot
        metrics: Optional metrics to display
        max_arrows: Maximum number of arrows to display
    """
    
    # Convert data to numpy arrays for vectorized operations
    gps_timestamps = np.array([entry['timestamp'] for entry in gps_data])
    gps_lats = np.array([entry['latitude'] for entry in gps_data])
    gps_lons = np.array([entry['longitude'] for entry in gps_data])
    
    valid_heading_data = [entry for entry in heading_data if heading_data_key_name in entry]
    if len(valid_heading_data) == 0:
        print(f"Warning: No matching data points found for {title}")
        return
    
    heading_timestamps = np.array([entry['timestamp'] for entry in valid_heading_data])
    heading_values = np.array([entry[heading_data_key_name] for entry in valid_heading_data])
    
    # Fully vectorized matching approach
    threshold = 0.8  # seconds
    
    # Create a 2D array of time differences between each heading timestamp and all GPS timestamps
    # shape: (n_headings, n_gps)
    time_diff_matrix = np.abs(heading_timestamps[:, np.newaxis] - gps_timestamps[np.newaxis, :])
    
    # Find the index of the closest GPS point for each heading
    closest_indices = np.argmin(time_diff_matrix, axis=1)
    
    # Get the minimum time difference for each heading
    min_diffs = np.min(time_diff_matrix, axis=1)

    # Create a mask for headings that have a GPS point within the threshold
    valid_mask = min_diffs < threshold
    
    # Extract timestamps, positions, and headings for valid matches
    valid_timestamps = heading_timestamps[valid_mask]
    valid_lats = gps_lats[closest_indices[valid_mask]]
    valid_lons = gps_lons[closest_indices[valid_mask]]
    valid_headings = heading_values[valid_mask]
    
    # Create a sorted index based on timestamps
    sorted_indices = np.argsort(valid_timestamps)
    
    # Sort all arrays using the timestamp order
    valid_timestamps = valid_timestamps[sorted_indices]
    valid_lats = valid_lats[sorted_indices]
    valid_lons = valid_lons[sorted_indices]
    valid_headings = valid_headings[sorted_indices]
    
    if len(valid_timestamps) == 0:
        print(f"Warning: No matching data points found for {title}")
        # Create a blank image with error message
        plt.figure(figsize=(12, 10))
        plt.text(0.5, 0.5, f"No matching data points found for {title}", 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        if output_path:
            plt.savefig(output_path, dpi=300)
        plt.close()
        return
    
    # Sparsify if too many points (vectorized)
    if len(valid_lats) > max_arrows:
        step = len(valid_lats) // max_arrows
        indices = np.arange(0, len(valid_lats), step)
        valid_lats = valid_lats[indices]
        valid_lons = valid_lons[indices]
        valid_headings = valid_headings[indices]
    
    # Convert heading to direction components (vectorized)
    # u = east component, v = north component
    headings_rad = np.radians(valid_headings)
    u = np.sin(headings_rad)
    v = np.cos(headings_rad)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot GPS track
    all_lats = [entry['latitude'] for entry in gps_data]
    all_lons = [entry['longitude'] for entry in gps_data]
    # plt.plot(all_lons, all_lats, 'b-', alpha=0.3, linewidth=1.5, label='GPS Track')

    # plot as a scatter plot
    plt.scatter(all_lons, all_lats, color='blue', s=1, label='GPS Track')
    
    # Plot heading arrows
    plt.quiver(valid_lons, valid_lats, u, v, color='red', scale=25, width=0.003, 
              alpha=0.8, label='Heading', pivot='mid')
    
    # Mark start and end points
    plt.scatter(all_lons[0], all_lats[0], color='green', s=100, label='Start')
    plt.scatter(all_lons[-1], all_lats[-1], color='red', s=100, label='End')
    
    # Build title with metrics if provided
    if metrics:
        # Handle potentially NaN or missing metrics
        mean_error = metrics.get('mean_error', float('nan'))
        median_error = metrics.get('median_error', float('nan'))
        max_error = metrics.get('max_error', float('nan'))
        
        # Format as strings with error handling
        mean_str = f"{mean_error:.2f}°" if not math.isnan(mean_error) else "N/A"
        median_str = f"{median_error:.2f}°" if not math.isnan(median_error) else "N/A"
        max_str = f"{max_error:.2f}°" if not math.isnan(max_error) else "N/A"
        
        metrics_text = f"\nMean Error: {mean_str}, Median: {median_str}, Max: {max_str}"
        title = title + metrics_text
    
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        # Show the plot if no output path is provided
        plt.show(block=True)
        
    plt.close()
    
def visualize_gps_track(data,output_path=None,title="gps track"):
    """
    Visualize GPS track data
    """
    valid_data = [entry for entry in data \
        if 'latitude' in entry and 'longitude' in entry\
            and entry['latitude'] is not None and entry['longitude'] is not None]
    if len(valid_data) == 0:
        print(f"Warning: No matching data points found for {title}")
        return
    
    lats = np.array([entry['latitude'] for entry in valid_data])
    lons = np.array([entry['longitude'] for entry in valid_data])
    
    plt.figure(figsize=(12, 10))
    plt.scatter(lons, lats, color='blue', s=1, label='GPS Track')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show(block=True)
    
    plt.close()
    
    
    
def visualize_speed(data,key="speed",output_path=None,title="Speed",metrics=None):
    """
    Visualize speed data
    """
    valid_data = [entry for entry in data if key in entry]
    if len(valid_data) == 0:
        print(f"Warning: No matching data points found for {title}")
        return
    
    timestamps = np.array([entry['timestamp'] for entry in valid_data])
    values = np.array([entry[key] for entry in valid_data])
    plt.figure(figsize=(12, 10))
    plt.plot(timestamps, values, 'b-', alpha=0.3, linewidth=1.5, label='Speed')
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel('Speed (m/s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show(block=True)
    
    plt.close()
    

class CompassCalibrator:
    """
    A class to calibrate compass (magnetometer) data using multiple methods:
    1. Ellipsoid fitting for hard/soft iron calibration
    2. GPS ground truth validation and calibration
    3. Wheel encoder (RPM) data integration
    
    This calibrator handles:
    - Time alignment between GPS, magnetometer, and RPM data
    - Calculation of ground truth headings from GPS data
    - Detection and filtering of magnetic anomalies
    - Outlier removal
    - Calibration to compensate for hard and soft iron effects
    - Kalman filtering with sensor fusion for smoothed outputs
    """
    
    def __init__(self, min_speed_threshold=0.5, max_heading_change_rate=45.0, 
                 outlier_threshold=3.0, time_alignment_threshold=0.5,
                 use_kalman_filter=False, wheel_diameter=130, wheel_base=200,
                 use_ellipsoid_fit=True, use_gps_calibration=True,
                 use_static_calibration=False, outlier_percentage=0.02):
        """
        Initialize the compass calibrator
        
        Args:
            min_speed_threshold: Minimum speed (m/s) to consider GPS headings reliable
            max_heading_change_rate: Maximum heading change rate (deg/s) for outlier detection
            outlier_threshold: Threshold in standard deviations for outlier detection
            time_alignment_threshold: Maximum time difference (s) for timestamp alignment
            use_kalman_filter: Whether to apply Kalman filtering for smoothing
            wheel_diameter: Diameter of the wheels in mm
            wheel_base: Distance between left and right wheels in mm
            use_ellipsoid_fit: Whether to use ellipsoid fitting for calibration
            use_gps_calibration: Whether to use GPS data for calibration
            use_static_calibration: Whether to use simple min/max calibration method
            outlier_percentage: Percentage of data to consider as outliers (0.02 = 2%)
        """
        self.min_speed_threshold = min_speed_threshold
        self.max_heading_change_rate = max_heading_change_rate
        self.outlier_threshold = outlier_threshold
        self.time_alignment_threshold = time_alignment_threshold
        self.use_kalman_filter = use_kalman_filter
        self.use_ellipsoid_fit = use_ellipsoid_fit
        self.use_gps_calibration = use_gps_calibration
        self.use_static_calibration = use_static_calibration
        self.outlier_percentage = outlier_percentage
        # Wheel parameters for RPM calculations (convert to meters)
        self.wheel_diameter = wheel_diameter / 1000.0  # mm to m
        self.wheel_base = wheel_base / 1000.0  # mm to m
        self.wheel_radius = self.wheel_diameter / 2.0
        self._kf_use_gps = False
        self._kf_use_rpm = False
        
        # Calibration parameters
        self.ellipsoid_params = None  # From ellipsoid fitting
        self.gps_cal_params = None    # From GPS ground truth
        self.static_cal_params = None  # From static min/max calibration
        self.validation_metrics = None
    

    @staticmethod
    def read_csv_data(file_path, data_type):
        """
        Read data from CSV file
        
        Args:
            file_path: Path to the CSV file
            data_type: Type of data ('gps', 'rpm', or 'imu')
            
        Returns:
            List of dictionaries with the data
        """
        data = []
        
        if data_type == 'imu':
            # Handle IMU data with nested JSON-like structure
            
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Skip header row
                
                for row in reader:
                    if len(row) < 4:  # Ensure we have enough columns
                        continue
                    
                    
                    # Extract compass data (magnetometer)
                    compass_str = row[0]
                    try:
                        # Clean up the string to make it valid JSON
                        compass_str = compass_str.replace('""', '"')  # Fix double quotes
                        # Extract the array content using regex
                        match = re.search(r'\[\[(.*?)\]\]', compass_str)
                        if match:
                            compass_values = match.group(1).split(', ')
                            # Remove remaining quotes
                            x = float(compass_values[0].replace('"', ''))
                            y = float(compass_values[1].replace('"', ''))
                            z = float(compass_values[2].replace('"', ''))
                            timestamp = float(compass_values[3].replace('"', ''))  # Last column is timestamp
                            
                            data.append({
                                'timestamp': timestamp,
                                'x': x,
                                'y': y,
                                'z': z
                            })
                    except Exception as e:
                        print(f"Error parsing compass data: {e}")
                        continue
        else:
            # Handle standard CSV formats
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    entry = {}
                    
                    if data_type == 'gps':
                        entry['latitude'] = float(row['latitude'])
                        entry['longitude'] = float(row['longitude'])
                        entry['timestamp'] = float(row['timestamp'])/1000

                    elif data_type == 'rpm':
                        entry['rpm_1'] = float(row['rpm_1'])
                        entry['rpm_2'] = float(row['rpm_2'])
                        entry['rpm_3'] = float(row['rpm_3'])
                        entry['rpm_4'] = float(row['rpm_4'])
                        entry['timestamp'] = float(row['timestamp'])/1000
                        
                    data.append(entry)
        
        return sorted(data, key=lambda x: x['timestamp'])
    def calibrate(self, gps_data, mag_data, rpm_data=None):
        """
        Main calibration function
        
        Args:
            gps_data: List of dictionaries with keys 'latitude', 'longitude', 'timestamp'
            mag_data: List of dictionaries with keys 'x', 'y', 'z', 'timestamp'
            rpm_data: List of dictionaries with keys 'rpm_1', 'rpm_2', 'rpm_3', 'rpm_4', 'timestamp'
            
        Returns:
            List of dictionaries with keys 'timestamp' and 'heading' (in degrees, NED frame)
            Important: the heading is from north to east, clockwise, in degrees
        """
        if isinstance(gps_data, str):
            gps_data = self.read_csv_data(gps_data, 'gps')
        if isinstance(mag_data, str):
            mag_data = self.read_csv_data(mag_data, 'imu')
        if isinstance(rpm_data, str):
            rpm_data = self.read_csv_data(rpm_data, 'rpm')
            
        # Step 1: Perform ellipsoid fitting on magnetometer data
        if self.use_ellipsoid_fit:
            self.ellipsoid_params = self._perform_ellipsoid_fitting(mag_data, gps_data)
        
        # Step 1b: Perform static calibration if enabled
        if self.use_static_calibration:
            self.static_cal_params = self._perform_static_calibration(mag_data)
        
        # apply ellipsoid calibration to mag_data
        if self.ellipsoid_params and self.use_ellipsoid_fit:
            mag_data_calibrated = self._apply_ellipsoid_calibration(mag_data)
            # disable ellipsoid fit to avoid double calibration
            self.use_ellipsoid_fit = False
        
        else:
            mag_data_calibrated = mag_data
        
        # Step 2: Preprocess and align data
        aligned_data = self._align_timestamps(gps_data, mag_data_calibrated, rpm_data)
        
        # Step 3: Calculate GPS headings and speeds
        aligned_data = self._calculate_gps_headings(aligned_data)
        
        # Step 4: Calculate RPM-based angular velocity (if available)
        if rpm_data:
            aligned_data = self._calculate_rpm_angular_velocity(aligned_data)
        
        # Step 5: Detect magnetic anomalies
        aligned_data = self._detect_magnetic_anomalies(aligned_data)
        
        # Step 6: Filter out data points where GPS heading is unreliable or magnetic anomalies exist
        filtered_data = [entry for entry in aligned_data 
                         if entry.get('speed', 0) >= self.min_speed_threshold and 
                         not entry.get('is_magnetic_anomaly', False)]
        
        # Step 7: Calculate raw magnetometer headings
        filtered_data = self._calculate_mag_headings(filtered_data)
        
        # Step 8: Remove outliers based on heading differences
        cleaned_data = self._remove_heading_outliers(filtered_data)
        
        # Step 9: Perform fine-tuning calibration with GPS if we have enough clean data
        if len(cleaned_data) >= 10:
            self.gps_cal_params = self._perform_gps_calibration(cleaned_data)
            # Use combined calibration parameters
            self.calibration_params = self._combine_calibration_params()
        else:
            print(f"Warning: Not enough clean data points for GPS calibration ({len(cleaned_data)} < 10)")
            # Use only ellipsoid calibration parameters
            self.calibration_params = self.ellipsoid_params
            
        # Step 10: Apply final calibration to all magnetometer readings
        calibrated_headings = self._apply_calibration(aligned_data)
        
        # Step 11: Apply Kalman filter for smoothing (if enabled)
        if self.use_kalman_filter:
            calibrated_headings = self._apply_kalman_filter(aligned_data, calibrated_headings)
        
        # Step 12: Validate calibration
        # self.validation_metrics = self._validate_calibration(aligned_data, calibrated_headings)
        
        return calibrated_headings
    
    def _get_ellipsoid_params(self, A, offset):
        """
        Extract ellipsoid parameters from matrix form
        
        Args:
            A: 3x3 quadratic form matrix
            offset: center of ellipsoid
            
        Returns:
            T: transformation matrix, radii: squared radii of ellipsoid
        """
        # Get eigenvalues and eigenvectors
        eigvals, eigvecs = eigh(A)
        
        # Handle negative eigenvalues - take absolute values
        # This makes the fit more like an ellipsoid even if the data suggests a hyperboloid
        eigvals = np.abs(eigvals)
        
        # Add a small epsilon to avoid division by very small numbers
        epsilon = 1e-10
        safe_eigvals = np.maximum(eigvals, epsilon)
        
        # Radii are inversely proportional to the square root of eigenvalues
        radii = 1.0 / safe_eigvals
        
        # Transform matrix from ellipsoid to sphere
        T = eigvecs
        
        return T, radii

    def _perform_ellipsoid_fitting(self, mag_data, gps_data=None):
        """
        Perform ellipsoid fitting to calibrate magnetometer data
        
        Args:
            mag_data: List of magnetometer readings with keys 'x', 'y', 'z', 'timestamp'
            gps_data: Optional list of GPS readings with keys 'latitude', 'longitude', 'timestamp'
                
        Returns:
            Dictionary with calibration parameters
        """
        try:
            # Extract magnetic field readings
            x = np.array([entry['x'] for entry in mag_data])
            y = np.array([entry['y'] for entry in mag_data])
            z = np.array([entry['z'] for entry in mag_data])
            
            # Determine the expected field strength (F)
            F = self._get_field_strength(gps_data)
            
            # Prepare data for ellipsoid fit - format as [3,N] array
            s = np.vstack([x, y, z])
            
            # Perform ellipsoid fit
            M, n, d = self._ellipsoid_fit(s)
            
            # Calculate calibration parameters
            try:
                # Get the inverse of M
                M_1 = scipy.linalg.inv(M)
                
                # Calculate the offset (hard iron bias)
                offset = -np.dot(M_1, n)
                
                # Calculate the transformation matrix (soft iron correction)
                # Note: F is already in raw LSB units matching the sensor data
                transform = np.real(F / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) * scipy.linalg.sqrtm(M))
                
                return {
                    'offset_x': float(offset[0]),
                    'offset_y': float(offset[1]),
                    'offset_z': float(offset[2]),
                    'transform': transform.tolist(),
                    'method': 'ellipsoid',
                    'field_strength': float(F)
                }
            except scipy.linalg.LinAlgError as e:
                print(f"Matrix inversion error: {e}")
                return {
                    'offset_x': 0.0,
                    'offset_y': 0.0,
                    'offset_z': 0.0,
                    'transform': np.eye(3).tolist(),
                    'method': 'identity',
                    'field_strength': float(F)
                }
                
        except Exception as e:
            print(f"Ellipsoid fitting failed: {e}")
            return {
                'offset_x': 0.0,
                'offset_y': 0.0,
                'offset_z': 0.0,
                'transform': np.eye(3).tolist(),
                'method': 'identity',
                'field_strength': 1000.0  # Default field strength
            }

    def _get_field_strength(self, gps_data, gain=3000):
        """
        Get the expected magnetic field strength in raw LSB units
        
        Args:
            gps_data: List of GPS data entries with latitude and longitude
            gain: Sensor gain in LSB/Gauss (default: 1090 for HMC5883L at ±1.3Ga)
            
        Returns:
            Field strength in raw LSB units
        """
        # Default field strength if we can't calculate it
        default_F = 1000.0
        
        # If no GPS data is provided, use default
        if not gps_data:
            print("No GPS data provided. Using default field strength.")
            return default_F
        
        try:
            # Find the first valid GPS reading
            valid_gps = next((entry for entry in gps_data 
                            if 'latitude' in entry and 'longitude' in entry 
                            and entry['latitude'] is not None 
                            and entry['longitude'] is not None), None)
            
            if not valid_gps:
                print("No valid GPS coordinates found. Using default field strength.")
                return default_F
            
            # Extract coordinates
            latitude = valid_gps['latitude']
            longitude = valid_gps['longitude']
            altitude = valid_gps.get('altitude', 0)
            
            # Use geomag to get the field strength
            try:
                geo_mag = geomag.geomag.GeoMag()
                result = geo_mag.GeoMag(latitude, longitude, altitude)
                total_field_nT = result.ti

                # Convert nT to Gauss
                total_field_gauss = total_field_nT * 1e-5
                
                # Convert Gauss to raw LSB value using gain
                raw_field = total_field_gauss * gain
                
                print(f"Magnetic field at ({latitude}, {longitude}): {total_field_nT} nT")
                print(f"Converted to raw LSB value: {raw_field:.2f}")
                
                return raw_field
            except ImportError:
                print("geomag library not found. Using default field strength.")
                return default_F
        except Exception as e:
            print(f"Error calculating field strength: {e}")
            return default_F

    def _ellipsoid_fit(self, s):
        """
        Estimate ellipsoid parameters from a set of points.
        
        Parameters
        ----------
        s : array_like
        The samples (M,N) where M=3 (x,y,z) and N=number of samples.
        
        Returns
        -------
        M, n, d : array_like, array_like, float
        The ellipsoid parameters M, n, d.
        """
        # D (samples)
        D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                    2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                    2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

        # S, S_11, S_12, S_21, S_22 (eq. 11)
        S = np.dot(D, D.T)
        S_11 = S[:6,:6]
        S_12 = S[:6,6:]
        S_21 = S[6:,:6]
        S_22 = S[6:,6:]

        # C (Eq. 8, k=4)
        C = np.array([[-1,  1,  1,  0,  0,  0],
                    [ 1, -1,  1,  0,  0,  0],
                    [ 1,  1, -1,  0,  0,  0],
                    [ 0,  0,  0, -4,  0,  0],
                    [ 0,  0,  0,  0, -4,  0],
                    [ 0,  0,  0,  0,  0, -4]])

        # v_1 (eq. 15, solution)
        E = np.dot(scipy.linalg.inv(C),
                S_11 - np.dot(S_12, np.dot(scipy.linalg.inv(S_22), S_21)))

        E_w, E_v = scipy.linalg.eig(E)

        v_1 = E_v[:, np.argmax(E_w)]
        if v_1[0] < 0: v_1 = -v_1

        # v_2 (eq. 13, solution)
        v_2 = np.dot(np.dot(-scipy.linalg.inv(S_22), S_21), v_1)

        # quadratic-form parameters, parameters h and f swapped as per correction 
        M = np.array([[v_1[0], v_1[5], v_1[4]],
                    [v_1[5], v_1[1], v_1[3]],
                    [v_1[4], v_1[3], v_1[2]]])
        n = np.array([[v_2[0]],
                    [v_2[1]],
                    [v_2[2]]])
        d = v_2[3]

        return M, n, d

    def _perform_ellipsoid_fitting_v0(self, mag_data):
        """
        Perform ellipsoid fitting to calibrate magnetometer data
        
        Args:
            mag_data: List of magnetometer readings
            
        Returns:
            Dictionary with calibration parameters
        """
        # Extract magnetic field readings
        x = np.array([entry['x'] for entry in mag_data])
        y = np.array([entry['y'] for entry in mag_data])
        z = np.array([entry['z'] for entry in mag_data])
        
        # Stack data
        D = np.column_stack((x**2, y**2, z**2, 2*y*z, 2*x*z, 2*x*y, 2*x, 2*y, 2*z, np.ones_like(x)))
        
        # Solve least squares for ellipsoid parameters
        try:
            # S is the solution to the least squares problem
            S, residuals, rank, singular_values = np.linalg.lstsq(D, np.ones_like(x), rcond=None)
            
            # Extract the ellipsoid parameters
            A = np.array([
                [S[0], S[5], S[4]],
                [S[5], S[1], S[3]],
                [S[4], S[3], S[2]]
            ])
            
            # Center (hard iron offset)
            v = np.array([S[6], S[7], S[8]])
            
            # Check if A is invertible
            try:
                A_inv = np.linalg.inv(A)
                offset = -A_inv.dot(v) / 2
            except np.linalg.LinAlgError:
                # If A is not invertible, use pseudo-inverse
                A_pinv = np.linalg.pinv(A)
                offset = -A_pinv.dot(v) / 2
            
            # Get the radii and transformation matrix
            T, radii = self._get_ellipsoid_params(A, offset)
            
            # Ensure radii is positive for sqrt calculation
            radii_sqrt = np.sqrt(np.abs(radii))
            
            # Normalize to get a sphere (avoid division by zero)
            epsilon = 1e-10
            R = np.diag(1.0 / np.maximum(radii_sqrt, epsilon))
            
            # Combined transformation
            W = R.dot(T)
            
            return {
                'offset_x': float(offset[0]),
                'offset_y': float(offset[1]),
                'offset_z': float(offset[2]),
                'transform': W.tolist(),
                'method': 'ellipsoid'
            }
        except Exception as e:
            print(f"Ellipsoid fitting failed: {e}")
            # Return identity parameters if fitting fails
            return {
                'offset_x': 0.0,
                'offset_y': 0.0,
                'offset_z': 0.0,
                'transform': np.eye(3).tolist(),
                'method': 'identity'
            }
    
    def _apply_ellipsoid_calibration(self, mag_data):
        """
        Apply ellipsoid calibration to magnetometer data
        
        Args:
            mag_data: List of magnetometer readings
            
        Returns:
            Calibrated magnetometer data
        """
        if self.ellipsoid_params is None:
            return mag_data
            
        offset_x = self.ellipsoid_params['offset_x']
        offset_y = self.ellipsoid_params['offset_y']
        offset_z = self.ellipsoid_params['offset_z']
        transform = np.array(self.ellipsoid_params['transform'])
        
        calibrated_data = []
        
        for entry in mag_data:
            # Center the data (remove hard iron)
            x_centered = entry['x'] - offset_x
            y_centered = entry['y'] - offset_y
            z_centered = entry['z'] - offset_z
            
            # Apply transformation (correct soft iron)
            v = np.array([x_centered, y_centered, z_centered])
            calibrated_v = transform.dot(v)
            
            # Create new entry with calibrated values
            new_entry = entry.copy()
            new_entry['x'] = float(calibrated_v[0])
            new_entry['y'] = float(calibrated_v[1])
            new_entry['z'] = float(calibrated_v[2])
            
            calibrated_data.append(new_entry)
            
        return calibrated_data
    
    def _align_timestamps(self, gps_data, mag_data, rpm_data=None):
        """
        Align GPS, magnetometer, and RPM data based on timestamps using interpolation
        
        Returns:
            List of dictionaries with aligned data
        """
        # Sort data by timestamp
        gps_data = sorted(gps_data, key=lambda x: x['timestamp'])
        mag_data = sorted(mag_data, key=lambda x: x['timestamp'])
        
        if rpm_data:
            rpm_data = sorted(rpm_data, key=lambda x: x['timestamp'])
        
        # For each magnetometer reading, find or interpolate GPS and RPM data
        aligned_data = []
        
        for mag_entry in mag_data:
            mag_ts = mag_entry['timestamp']
            
            # Find or interpolate GPS data
            gps_lat, gps_lon, is_gps_interpolated = self._interpolate_data_point(
                gps_data, mag_ts, 'latitude', 'longitude')
            
            # Prepare aligned entry
            aligned_entry = {
                'timestamp': mag_ts,
                'latitude': gps_lat,
                'longitude': gps_lon,
                'mag_x': mag_entry['x'],
                'mag_y': mag_entry['y'],
                'mag_z': mag_entry['z'],
                'is_interpolated_gps': is_gps_interpolated
            }
            
            # Add RPM data if available
            if rpm_data:
                rpm_values, is_rpm_interpolated = self._interpolate_rpm_data(rpm_data, mag_ts)
                aligned_entry.update(rpm_values)
                aligned_entry['is_interpolated_rpm'] = is_rpm_interpolated
                
            aligned_data.append(aligned_entry)
        
        return aligned_data
    
    def _interpolate_data_point(self, data_list, target_ts, *fields):
        """
        Interpolate data fields at a target timestamp
        
        Args:
            data_list: List of data points
            target_ts: Target timestamp
            fields: Fields to interpolate
            
        Returns:
            Tuple of interpolated values and boolean indicating if interpolation was used
        """
        # Find surrounding data points
        lower_point = None
        upper_point = None
        
        for point in data_list:
            if point['timestamp'] <= target_ts:
                lower_point = point
            elif point['timestamp'] > target_ts:
                upper_point = point
                break
        
        # Check if we have enough data for interpolation
        is_interpolated = True
        
        if lower_point is None and upper_point is None:
            # No data points available
            return tuple([None] * len(fields) + [True])
            
        elif lower_point is None:
            # Only have data after target time
            if upper_point['timestamp'] - target_ts < self.time_alignment_threshold:
                # Close enough to use the upper point
                return tuple([upper_point[field] for field in fields] + [True])
            else:
                # Too far to use
                return tuple([None] * len(fields) + [True])
                
        elif upper_point is None:
            # Only have data before target time
            if target_ts - lower_point['timestamp'] < self.time_alignment_threshold:
                # Close enough to use the lower point
                return tuple([lower_point[field] for field in fields] + [True])
            else:
                # Too far to use
                return tuple([None] * len(fields) + [True])
                
        elif lower_point['timestamp'] == target_ts:
            # Exact match with lower point
            return tuple([lower_point[field] for field in fields] + [False])
            
        elif upper_point['timestamp'] == target_ts:
            # Exact match with upper point
            return tuple([upper_point[field] for field in fields] + [False])
            
        else:
            # Perform linear interpolation
            lower_ts = lower_point['timestamp']
            upper_ts = upper_point['timestamp']
            
            # Check if time difference is too large
            if upper_ts - lower_ts > self.time_alignment_threshold * 10:
                # Too sparse data
                if target_ts - lower_ts < self.time_alignment_threshold:
                    return tuple([lower_point[field] for field in fields] + [True])
                elif upper_ts - target_ts < self.time_alignment_threshold:
                    return tuple([upper_point[field] for field in fields] + [True])
                else:
                    return tuple([None] * len(fields) + [True])
            
            # Calculate interpolation weight
            weight = (target_ts - lower_ts) / (upper_ts - lower_ts)
            
            # Interpolate values
            values = []
            for field in fields:
                interpolated = lower_point[field] + weight * (upper_point[field] - lower_point[field])
                values.append(interpolated)
                
            return tuple(values + [True])
    
    def _interpolate_rpm_data(self, rpm_data, target_ts):
        """
        Interpolate RPM data at a target timestamp
        
        Args:
            rpm_data: List of RPM readings
            target_ts: Target timestamp
            
        Returns:
            Dictionary with RPM values and boolean indicating if interpolation was used
        """
        fields = ['rpm_1', 'rpm_2', 'rpm_3', 'rpm_4']
        results = self._interpolate_data_point(rpm_data, target_ts, *fields)
        
        values = results[:-1]  # All but the last element (is_interpolated)
        is_interpolated = results[-1]
        
        # Return as dictionary
        if all(v is not None for v in values):
            return {field: value for field, value in zip(fields, values)}, is_interpolated
        else:
            return {}, True
    
    def _calculate_gps_headings(self, aligned_data):
        """
        Calculate headings from GPS data
        
        Returns:
            Data with GPS headings and speeds added
        """
        # Define minimum distance threshold to filter noise
        min_distance_threshold = 0.1  # meters

        if USE_NED:
            # Convert all GPS data to UTM coordinates
            for entry in aligned_data:
                if 'latitude' in entry and 'longitude' in entry\
                    and entry['latitude'] is not None and entry['longitude'] is not None:
                    east, north, _, _ = utm.from_latlon(entry['latitude'], entry['longitude'])
                    entry['longitude'] = east
                    entry['latitude'] = north
        
        for i in range(1, len(aligned_data)):
            prev_entry = aligned_data[i-1]
            curr_entry = aligned_data[i]
            
            # Skip only if GPS data is missing, but ALLOW interpolated data
            if ('latitude' not in prev_entry or 'longitude' not in prev_entry or
                'latitude' not in curr_entry or 'longitude' not in curr_entry or
                prev_entry['latitude'] is None or prev_entry['longitude'] is None or
                curr_entry['latitude'] is None or curr_entry['longitude'] is None):
                continue
            
            # Calculate time difference
            time_diff = curr_entry['timestamp'] - prev_entry['timestamp']
            if time_diff <= 0:
                continue
                
            if USE_NED:
                # Convert to UTM coordinates
                prev_e, prev_n = prev_entry['longitude'], prev_entry['latitude']
                curr_e, curr_n = curr_entry['longitude'], curr_entry['latitude']
                
                # Calculate distance in meters
                distance = math.sqrt((curr_e - prev_e) ** 2 + (curr_n - prev_n) ** 2)

                # Skip if distance is too small (noise)
                if distance < 0.2:
                    continue
                
                # Calculate heading
                # TODO: what is the correct way to do this?
                heading = math.atan2(curr_e - prev_e, curr_n - prev_n)
                heading = math.degrees(heading)
                heading = (heading + 360) % 360

            else:
                # Convert to radians
                lat1, lon1 = math.radians(prev_entry['latitude']), math.radians(prev_entry['longitude'])
                lat2, lon2 = math.radians(curr_entry['latitude']), math.radians(curr_entry['longitude'])
                
                # Calculate distance using haversine formula
                a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                distance = 6371000 * c  # Earth radius in meters
                
                # Skip if distance is too small (noise)
                if distance < min_distance_threshold:
                    continue
                
                # Calculate heading
                y = math.sin(lon2 - lon1) * math.cos(lat2)
                x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
                heading = math.atan2(y, x)
                
                # Convert to degrees and normalize to 0-360
                heading = math.degrees(heading)
                heading = (heading + 360) % 360
            
            # Calculate speed
            speed = distance / time_diff
            
            # Store heading and speed
            curr_entry['gps_heading'] = heading
            curr_entry['speed'] = speed
        
        return aligned_data
    # def _calculate_gps_headings(self, aligned_data):
    #     """
    #     Calculate headings from GPS data using easting/northing coordinates
        
    #     Returns:
    #         Data with GPS headings and speeds added
    #     """
    #     # Define minimum distance threshold to filter noise
    #     min_distance_threshold = 0.1  # meters
        
    #     # Set up a projection if not already done
    #     if not hasattr(self, 'proj') or self.proj is None:
    #         # Get reference point from first valid GPS entry
    #         for entry in aligned_data:
    #             if 'latitude' in entry and 'longitude' in entry and entry['latitude'] is not None:
    #                 ref_lat = entry['latitude']
    #                 ref_lon = entry['longitude']
                    
    #                 # Set up a transverse Mercator projection centered on the reference point
    #                 import pyproj
    #                 proj_string = f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    #                 self.proj = pyproj.Proj(proj_string)
    #                 break
        
    #     # Calculate headings
    #     for i in range(1, len(aligned_data)):
    #         prev_entry = aligned_data[i-1]
    #         curr_entry = aligned_data[i]
            
    #         # Skip if GPS data is missing
    #         if ('latitude' not in prev_entry or 'longitude' not in prev_entry or
    #             'latitude' not in curr_entry or 'longitude' not in curr_entry or
    #             prev_entry['latitude'] is None or prev_entry['longitude'] is None or
    #             curr_entry['latitude'] is None or curr_entry['longitude'] is None):
    #             continue
            
    #         # Calculate time difference
    #         time_diff = curr_entry['timestamp'] - prev_entry['timestamp']
    #         if time_diff <= 0:
    #             continue
                
    #         # Convert to easting/northing
    #         east1, north1 = self.proj(prev_entry['longitude'], prev_entry['latitude'])
    #         east2, north2 = self.proj(curr_entry['longitude'], curr_entry['latitude'])
            
    #         # Calculate distance in the projected coordinate system
    #         delta_east = east2 - east1
    #         delta_north = north2 - north1
    #         distance = math.sqrt(delta_east**2 + delta_north**2)
            
    #         # Skip if distance is too small (noise)
    #         if distance < min_distance_threshold:
    #             continue
            
    #         # Calculate heading using Cartesian coordinates
    #         # atan2(delta_east, delta_north) gives heading where 0° is North and 90° is East
    #         heading = math.degrees(math.atan2(delta_east, delta_north))
    #         heading = (heading + 360) % 360
            
    #         # Calculate speed
    #         speed = distance / time_diff
            
    #         # Store heading and speed
    #         curr_entry['gps_heading'] = heading
    #         curr_entry['speed'] = speed
        
    #     return aligned_data
    
    def _calculate_rpm_angular_velocity(self, aligned_data):
        """
        Calculate angular velocity from RPM data
        
        Returns:
            Data with RPM-based angular velocity added
        """
        for entry in aligned_data:
            # Skip if RPM data is missing
            if ('rpm_1' not in entry or 'rpm_2' not in entry or 
                'rpm_3' not in entry or 'rpm_4' not in entry):
                continue
                
            # Convert RPM to rad/s
            # For a differential drive, we use the average of left and right wheels
            rpm_left = (entry['rpm_1'] + entry['rpm_3']) / 2  # Front-Left and Rear-Left
            rpm_right = (entry['rpm_2'] + entry['rpm_4']) / 2  # Front-Right and Rear-Right
            
            # Convert RPM to linear velocity (m/s)
            v_left = (rpm_left / 60) * 2 * math.pi * self.wheel_radius
            v_right = (rpm_right / 60) * 2 * math.pi * self.wheel_radius
            
            # Calculate angular velocity (rad/s)
            angular_velocity =  (v_right - v_left) / self.wheel_base
            
            # Convert to deg/s
            angular_velocity_deg = math.degrees(angular_velocity)
            
            entry['rpm_angular_velocity'] = angular_velocity_deg
            
            # Calculate linear velocity (m/s)
            linear_velocity = (v_right + v_left) / 2
            entry['rpm_linear_velocity'] = linear_velocity
        
        # Calculate heading change based on angular velocity
        for i in range(1, len(aligned_data)):
            prev_entry = aligned_data[i-1]
            curr_entry = aligned_data[i]
            
            # Skip if angular velocity not available
            if ('rpm_angular_velocity' not in prev_entry or 
                'rpm_angular_velocity' not in curr_entry):
                continue
                
            # Calculate time difference
            time_diff = curr_entry['timestamp'] - prev_entry['timestamp']
            if time_diff <= 0:
                continue
            
            # Integrate angular velocity to get heading change
            avg_angular_vel = (prev_entry['rpm_angular_velocity'] + curr_entry['rpm_angular_velocity']) / 2
            heading_change = avg_angular_vel * time_diff
            
            # If previous entry has a heading, calculate current heading
            if 'rpm_heading' in prev_entry:
                curr_entry['rpm_heading'] = (prev_entry['rpm_heading'] + heading_change) % 360
            elif 'gps_heading' in prev_entry:
                # Initialize RPM heading with GPS heading if available
                curr_entry['rpm_heading'] = (prev_entry['gps_heading'] + heading_change) % 360
        
        return aligned_data
    
    def _detect_magnetic_anomalies(self, data):
        """
        Detect magnetic anomalies like proximity to metal structures
        
        Returns:
            Data with anomaly flags
        """
        # Extract magnetic field strengths
        field_strengths = []
        for entry in data:
            # Calculate total magnetic field strength
            strength = np.sqrt(entry['mag_x']**2 + entry['mag_y']**2 + entry['mag_z']**2)
            entry['field_strength'] = strength
            field_strengths.append(strength)
        
        if not field_strengths:
            return data
            
        # Calculate statistics
        mean_strength = np.mean(field_strengths)
        std_strength = np.std(field_strengths)
        
        # Flag anomalies
        for entry in data:
            # Check if field strength is anomalous
            is_anomaly = abs(entry['field_strength'] - mean_strength) > self.outlier_threshold * std_strength
            
            # Check if field strength is very low (near zero)
            is_weak = entry['field_strength'] < mean_strength * 0.2
            
            entry['is_magnetic_anomaly'] = is_anomaly or is_weak
        
        return data
    
    def _calculate_mag_headings(self, data):
        """
        Calculate headings from magnetometer data in the NED frame
        According to the provided schema:
        - Positive X = Vertical Up
        - Positive Y = West
        - Positive Z = South
        
        Converting to NED (North-East-Down) frame:
        - North = -Z (since Z points South)
        - East = -Y (since Y points West)
        - Down = -X (since X points Up)
        
        Returns:
            Data with magnetometer headings added
        """
        for entry in data:
            # TODO: what is the correct way to do this?
            north = -entry['mag_z']
            east = entry['mag_y']
            
            # Calculate heading (clockwise from North)
            heading = math.degrees(math.atan2(east, north))
            heading = (heading + 360) % 360
            
            entry['mag_heading'] = heading
        
        return data
    
    def _remove_heading_outliers(self, data):
        """
        Remove outliers based on heading differences and change rates
        
        Returns:
            Cleaned data without outliers
        """
        # Increase outlier threshold to allow more data
        local_outlier_threshold = 5.0  # Increased from 3.0
        local_max_change_rate = 90.0   # Increased from 45.0
        
        # Calculate heading differences and change rates
        heading_diffs = []
        change_rates = []
        
        for i in range(1, len(data)):
            if 'gps_heading' in data[i] and 'mag_heading' in data[i]:
                # Heading difference
                gps_heading = data[i]['gps_heading']
                mag_heading = data[i]['mag_heading']
                
                # Calculate smallest angle difference
                diff = (mag_heading - gps_heading + 180) % 360 - 180
                heading_diffs.append(abs(diff))
                data[i]['heading_diff'] = diff
                
                # Heading change rate
                if 'gps_heading' in data[i-1]:
                    time_diff = data[i]['timestamp'] - data[i-1]['timestamp']
                    if time_diff > 0:
                        prev_heading = data[i-1]['gps_heading']
                        curr_heading = data[i]['gps_heading']
                        
                        # Calculate smallest angle difference for change
                        heading_change = (curr_heading - prev_heading + 180) % 360 - 180
                        rate = abs(heading_change) / time_diff
                        
                        change_rates.append(rate)
                        data[i]['heading_change_rate'] = rate
        
        if not heading_diffs:
            print("Warning: No heading differences found, returning all data without outlier removal")
            return data
            
        # Calculate statistics for outlier detection
        mean_diff = np.mean(heading_diffs)
        std_diff = np.std(heading_diffs) if len(heading_diffs) > 1 else 1
        
        # Filter out outliers
        cleaned_data = []
        
        for entry in data:
            # Check heading difference outlier
            is_diff_outlier = False
            if 'heading_diff' in entry:
                is_diff_outlier = abs(entry['heading_diff']) > mean_diff + local_outlier_threshold * std_diff
            
            # Check heading change rate outlier
            is_rate_outlier = False
            if 'heading_change_rate' in entry:
                is_rate_outlier = entry['heading_change_rate'] > local_max_change_rate
            
            # Include only non-outliers
            if not is_diff_outlier and not is_rate_outlier:
                cleaned_data.append(entry)
        
        # Always return at least some data
        if len(cleaned_data) < 5 and len(data) > 0:
            print(f"Warning: Too few data points after outlier removal ({len(cleaned_data)} < 5), using all data")
            return data
        
        return cleaned_data
    
    def _perform_gps_calibration(self, data):
        # Extract data for calibration
        mag_headings = np.array([entry['mag_heading'] for entry in data])
        gps_headings = np.array([entry['gps_heading'] for entry in data])
        
        # More complex model with sinusoidal components to better model 
        # directional distortions (similar to soft iron effects)
        initial_params = [1.0, 0.0, 0.0, 0.0]  # [scale, offset, sin_amp, cos_amp]
        
        def residuals(params):
            scale, offset, sin_amp, cos_amp = params
            
            # Apply calibration with sinusoidal terms
            angle_rad = np.radians(mag_headings)
            correction = sin_amp * np.sin(angle_rad) + cos_amp * np.cos(angle_rad)
            calibrated_headings = (scale * mag_headings + offset + correction) % 360
            
            # Calculate errors
            errors = np.array([((c - g + 180) % 360 - 180) for c, g in zip(calibrated_headings, gps_headings)])
            return errors
        
        # Perform optimization
        try:
            result = least_squares(residuals, initial_params, loss='soft_l1')
            return {
                'scale': float(result.x[0]),
                'offset': float(result.x[1]),
                'sin_amp': float(result.x[2]),
                'cos_amp': float(result.x[3]),
                'method': 'gps'
            }
        except Exception as e:
            print(f"GPS calibration optimization failed: {e}")
            # Return default parameters in case of failure
            return {
                'scale': 1.0,
                'offset': 0.0,
                'sin_amp': 0.0,
                'cos_amp': 0.0,
                'method': 'default'
            }
    
    def _perform_static_calibration(self, mag_data):
        """
        Perform simple static calibration based on min/max values with outlier removal
        
        Args:
            mag_data: List of dictionaries with keys 'x', 'y', 'z', 'timestamp'
            
        Returns:
            Dictionary with offset and scale values
        """
        # Extract x, y, z values
        x_values = [entry['x'] for entry in mag_data]
        y_values = [entry['y'] for entry in mag_data]
        z_values = [entry['z'] for entry in mag_data]
        
        # Remove outliers if specified
        if self.outlier_percentage > 0:
            for values in [x_values, y_values, z_values]:
                values.sort()
                outlier_count = int(len(values) * self.outlier_percentage / 2)
                # Remove top and bottom outliers
                if outlier_count > 0:
                    values = values[outlier_count:-outlier_count]
        
        # Calculate offsets (hard iron correction)
        offset_x = (max(x_values) + min(x_values)) / 2
        offset_y = (max(y_values) + min(y_values)) / 2
        offset_z = (max(z_values) + min(z_values)) / 2
        
        # Calculate scales (soft iron correction)
        x_range = max(x_values) - min(x_values)
        y_range = max(y_values) - min(y_values)
        z_range = max(z_values) - min(z_values)
        
        # Calculate average range to normalize to
        avg_range = (x_range + y_range + z_range) / 3
        
        # Calculate scale factors (avoid division by zero)
        scale_x = avg_range / max(x_range, 0.001)
        scale_y = avg_range / max(y_range, 0.001)
        scale_z = avg_range / max(z_range, 0.001)
        
        print(f"Static calibration: offsets ({offset_x:.2f}, {offset_y:.2f}, {offset_z:.2f})")
        print(f"Static calibration: scales ({scale_x:.2f}, {scale_y:.2f}, {scale_z:.2f})")
        
        return {
            'offset_x': float(offset_x),
            'offset_y': float(offset_y),
            'offset_z': float(offset_z),
            'scale_x': float(scale_x),
            'scale_y': float(scale_y),
            'scale_z': float(scale_z),
            'method': 'static'
        }
    
    def _combine_calibration_params(self):
        """
        Combine ellipsoid, static and GPS calibration parameters
        
        Returns:
            Combined calibration parameters
        """
        combined = {}
        
        if self.use_ellipsoid_fit and self.ellipsoid_params is not None:
            combined['ellipsoid'] = self.ellipsoid_params
        
        if self.use_static_calibration and self.static_cal_params is not None:
            combined['static'] = self.static_cal_params
        
        if self.use_gps_calibration and self.gps_cal_params is not None:
            combined['gps'] = self.gps_cal_params
        
        combined['method'] = 'combined'
        return combined
    
    def _apply_calibration(self, data):
        """
        Apply calibration to all magnetometer readings
        
        Returns:
            List of dictionaries with keys 'timestamp' and 'heading'
        """
        if (not self.use_ellipsoid_fit and not self.use_gps_calibration and 
            not self.use_static_calibration):
            print("Warning: No calibration method enabled, using raw magnetometer data")
        
        calibrated_headings = []
        
        for entry in data:
            # Apply static calibration if enabled
            if self.use_static_calibration and self.static_cal_params is not None:
                offset_x = self.static_cal_params['offset_x']
                offset_y = self.static_cal_params['offset_y']
                offset_z = self.static_cal_params['offset_z']
                scale_x = self.static_cal_params['scale_x']
                scale_y = self.static_cal_params['scale_y']
                scale_z = self.static_cal_params['scale_z']
                
                # Apply hard iron correction (offset)
                x_centered = entry['mag_x'] - offset_x
                y_centered = entry['mag_y'] - offset_y
                z_centered = entry['mag_z'] - offset_z
                
                # Apply soft iron correction (scale)
                x_calibrated = x_centered * scale_x
                y_calibrated = y_centered * scale_y
                z_calibrated = z_centered * scale_z
                
                # Convert to NED frame
                north = -z_calibrated
                east = y_calibrated
                
            # Apply ellipsoid calibration if enabled
            elif self.use_ellipsoid_fit and self.ellipsoid_params is not None:
                offset_x = self.ellipsoid_params['offset_x']
                offset_y = self.ellipsoid_params['offset_y']
                offset_z = self.ellipsoid_params['offset_z']
                transform = np.array(self.ellipsoid_params['transform'])
                
                # Apply hard iron correction
                x_centered = entry['mag_x'] - offset_x
                y_centered = entry['mag_y'] - offset_y
                z_centered = entry['mag_z'] - offset_z
                
                # Apply soft iron correction
                v = np.array([x_centered, y_centered, z_centered])
                calibrated_v = transform.dot(v)

                # Convert to NED frame
                # TODO: what is the correct way to do this?
                north = -calibrated_v[2]  # North = -Z
                east = calibrated_v[1]   # East = -Y
                
            else:
                # Convert directly to NED frame
                # TODO: what is the correct way to do this?
                north = -entry['mag_z']
                east = entry['mag_y']
            

            heading = math.degrees(math.atan2(east, north))
            heading = (heading + 360) % 360
            
            # Apply GPS scale/offset calibration if available
            if self.use_gps_calibration and 'gps' in self.calibration_params:
                scale = self.calibration_params['gps']['scale']
                offset = self.calibration_params['gps']['offset']
                sin_amp = self.calibration_params['gps']['sin_amp']
                cos_amp = self.calibration_params['gps']['cos_amp']
                
                # Apply sinusoidal correction
                angle_rad = np.radians(heading)
                correction = sin_amp * np.sin(angle_rad) + cos_amp * np.cos(angle_rad)
                heading = (scale * heading + offset + correction) % 360
            
            calibrated_headings.append({
                'timestamp': entry['timestamp'],
                'heading': heading
            })
        
        # get numpy array of headings
        headings = np.array([entry['heading'] for entry in calibrated_headings])
        timestamps = np.array([entry['timestamp'] for entry in calibrated_headings])
        all = np.stack((headings, timestamps), axis=1)
        return all
    
    def _apply_kalman_filter(self, aligned_data, calibrated_headings):
        """
        Apply Kalman filtering to smooth the heading estimates
        
        Args:
            aligned_data: Original aligned data with RPM and GPS info
            calibrated_headings: Calibrated magnetometer headings
            
        Returns:
            Smoothed headings
        """
        if not calibrated_headings:
            return []
            
        # Convert to dict for easier lookup
        heading_dict = {entry['timestamp']: entry['heading'] for entry in calibrated_headings}
        
        # State vector: [heading, heading_rate]
        n_states = 2
        # Measurement vector: [mag_heading, gps_heading (optional), rpm_heading (optional)]
        n_measurements = 1  # Maximum number of measurements
        if self._kf_use_gps:
            n_measurements += 1
        if self._kf_use_rpm:
            n_measurements += 1
        
        # Initialize Kalman filter
        kf = KalmanFilter(dim_x=n_states, dim_z=n_measurements)
        
        # Initial state - use first heading
        first_ts = calibrated_headings[0]['timestamp']
        first_heading = calibrated_headings[0]['heading']
        kf.x = np.array([[first_heading], [0]])  # [heading, heading_rate]
        
        # Initial covariance
        kf.P = np.diag([100, 10])  # High uncertainty initially
        
        # Process noise (uncertainty in how heading evolves)
        kf.Q = np.diag([0.1, 1])  # Adjust based on expected system dynamics
        
        # Measurement noise (uncertainties in each sensor)
        # Order: [mag_heading, gps_heading (optional), rpm_heading (optional)]
        kf.R = np.diag([5, 10, 20])  # GPS more reliable than magnetometer, RPM less reliable
        
        smoothed_headings = []
        prev_ts = first_ts
        
        for i, entry in enumerate(aligned_data):
            current_ts = entry['timestamp']
            
            # Skip first entry as it was used for initialization
            if i == 0:
                smoothed_headings.append({
                    'timestamp': current_ts,
                    'heading': first_heading
                })
                continue
                    
            # Time update
            dt = current_ts - prev_ts
            if dt <= 0:
                # Skip entries with non-positive time difference
                continue
                    
            # Update state transition matrix for current dt
            kf.F = np.array([
                [1, dt],
                [0, 1]
            ])
            
            # Process noise increases with time
            kf.Q = np.array([
                [0.05*dt, 0.02*dt],
                [0.02*dt, 0.05*dt]
            ])
            
            # Predict
            kf.predict()
            
            # Prepare measurement and measurement matrix
            measurements = []
            H_rows = []
            
            # 1. Magnetometer heading (always available after calibration)
            if current_ts in heading_dict:
                mag_heading = heading_dict[current_ts]
                # Handle heading wraparound
                predicted_heading = kf.x[0, 0] % 360
                # Adjust measurement to minimize difference
                diff = (mag_heading - predicted_heading + 180) % 360 - 180
                adjusted_mag_heading = predicted_heading + diff
                measurements.append(adjusted_mag_heading)
                H_rows.append([1, 0])  # Measurement maps to heading state
            
            # 2. GPS heading (if available and reliable)
            if ('gps_heading' in entry and 
                entry.get('speed', 0) >= self.min_speed_threshold and
                self._kf_use_gps):
                gps_heading = entry['gps_heading']
                # Handle heading wraparound
                predicted_heading = kf.x[0, 0] % 360
                # Adjust measurement to minimize difference
                diff = (gps_heading - predicted_heading + 180) % 360 - 180
                adjusted_gps_heading = predicted_heading + diff
                measurements.append(adjusted_gps_heading)
                H_rows.append([1, 0])  # Measurement maps to heading state
            
            # 3. RPM heading (if available)
            if 'rpm_heading' in entry and self._kf_use_rpm:
                rpm_heading = entry['rpm_heading']
                # Handle heading wraparound
                predicted_heading = kf.x[0, 0] % 360
                # Adjust measurement to minimize difference
                diff = (rpm_heading - predicted_heading + 180) % 360 - 180
                adjusted_rpm_heading = predicted_heading + diff
                measurements.append(adjusted_rpm_heading)
                H_rows.append([1, 0])  # Measurement maps to heading state
            
            # 4. RPM angular velocity (if available)
            if 'rpm_angular_velocity' in entry and self._kf_use_rpm:
                rpm_angular_vel = entry['rpm_angular_velocity']
                measurements.append(rpm_angular_vel)
                H_rows.append([0, 1])  # Measurement maps to heading rate state
            
            # Update if we have measurements
            if measurements:
                # FIX: Ensure we have a correctly sized measurement vector
                # Create a zero vector with the expected shape (3, 1)
                z = np.zeros((n_measurements, 1))
                # Create a zero matrix with the expected shape
                H = np.zeros((n_measurements, n_states))
                # High uncertainty for padding values
                R = np.eye(n_measurements) * 15
                
                # Fill in available measurements (up to the expected number)
                for j in range(min(len(measurements), n_measurements)):
                    z[j, 0] = measurements[j]
                    H[j] = H_rows[j]
                    
                    
                    # Set appropriate uncertainty based on measurement type
                    if H_rows[j][0] == 1:  # Heading measurement
                        if j == 0:  # Magnetometer
                            R[j, j] = 10
                        elif j == 1 and self._kf_use_gps:  # GPS
                            gps_noise = 20.0 / max(1.0, entry.get('speed', 0))  # More reliable at higher speeds
                            R[j, j] = gps_noise
                        elif j == 2 and self._kf_use_rpm:  # RPM
                            R[j, j] = 20
                    elif H_rows[j][1] == 1:  # Angular velocity measurement
                        R[j, j] = 30
                
                # Update Kalman filter parameters
                kf.H = H
                kf.R = R
                
                # Update state with correctly shaped measurement vector
                kf.update(z)
            
            # Get smoothed heading (normalized to 0-360)
            smoothed_heading = kf.x[0, 0] % 360
            
            # Save result
            smoothed_headings.append({
                'timestamp': current_ts,
                'heading': smoothed_heading
            })
            
            # Update previous timestamp
            prev_ts = current_ts
        
        return smoothed_headings
    
    def _validate_calibration(self, original_data, calibrated_headings):
        """
        Validate the calibration by comparing with GPS headings
        
        Returns:
            Validation metrics
        """
        # Build dictionaries for quick lookups
        gps_dict = {}
        for entry in original_data:
            if 'gps_heading' in entry and entry.get('speed', 0) >= self.min_speed_threshold:
                gps_dict[entry['timestamp']] = entry['gps_heading']
        
        cal_dict = {entry['timestamp']: entry['heading'] for entry in calibrated_headings}
        
        # Calculate errors
        errors = []
        for ts in set(gps_dict.keys()).intersection(set(cal_dict.keys())):
            gps_heading = gps_dict[ts]
            cal_heading = cal_dict[ts]
            
            # Calculate smallest angle difference
            error = (cal_heading - gps_heading + 180) % 360 - 180
            errors.append(abs(error))
        
        if not errors:
            return {'error_count': 0, 'mean_error': float('nan'), 'max_error': float('nan')}
        
        return {
            'error_count': len(errors),
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors)
        }
    
    def get_calibration_params(self):
        """
        Get the calibration parameters
        
        Returns:
            Dictionary with calibration parameters
        """
        return {
            'ellipsoid': self.ellipsoid_params,
            'gps': self.gps_cal_params,
            'static': self.static_cal_params,
            'combined': self.calibration_params
        }
    
    def get_validation_metrics(self):
        """
        Get the validation metrics
        
        Returns:
            Dictionary with validation metrics
        """
        return self.validation_metrics
    
    def apply_calibration_to_new_data(self, mag_data, rpm_data=None):
        """
        Apply the existing calibration to new magnetometer data
        
        Args:
            mag_data: List of dictionaries with keys 'x', 'y', 'z', 'timestamp'
            rpm_data: Optional list of dictionaries with RPM data
            
        Returns:
            List of dictionaries with keys 'timestamp' and 'heading' (in degrees, NED frame)
        """
        if self.calibration_params is None:
            raise ValueError("Calibration must be performed before applying it to new data")
        
        # Apply ellipsoid calibration first
        mag_data_calibrated = self._apply_ellipsoid_calibration(mag_data)
        
        # Align with RPM data if provided
        if rpm_data:
            aligned_data = self._align_timestamps([], mag_data_calibrated, rpm_data)
            aligned_data = self._calculate_rpm_angular_velocity(aligned_data)
        else:
            # Create minimal aligned data with just mag readings
            aligned_data = []
            for entry in mag_data_calibrated:
                aligned_entry = {
                    'timestamp': entry['timestamp'],
                    'mag_x': entry['x'],
                    'mag_y': entry['y'],
                    'mag_z': entry['z']
                }
                aligned_data.append(aligned_entry)
        
        # Apply calibration
        calibrated_headings = self._apply_calibration(aligned_data)
        
        # Apply Kalman filter for smoothing (if enabled)
        if self.use_kalman_filter:
            calibrated_headings = self._apply_kalman_filter(aligned_data, calibrated_headings)
            
        return calibrated_headings


# Example usage
def example():
    # Sample data format
    gps_data = [
        {'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': 1609459200},
        {'latitude': 37.7750, 'longitude': -122.4195, 'timestamp': 1609459201},
        # More GPS data...
    ]
    
    mag_data = [
        {'x': 123, 'y': 456, 'z': 789, 'timestamp': 1609459200},
        {'x': 124, 'y': 457, 'z': 788, 'timestamp': 1609459201},
        # More magnetometer data...
    ]
    
    rpm_data = [
        {'rpm_1': 50, 'rpm_2': 50, 'rpm_3': 50, 'rpm_4': 50, 'timestamp': 1609459200},
        {'rpm_1': 55, 'rpm_2': 45, 'rpm_3': 55, 'rpm_4': 45, 'timestamp': 1609459201},
        # More RPM data...
    ]
    
    # Initialize calibrator with robot parameters
    calibrator = CompassCalibrator(wheel_diameter=130, wheel_base=200)
    
    # Perform calibration with all available data
    calibrated_headings = calibrator.calibrate(gps_data, mag_data, rpm_data)
    
    # Get calibration parameters
    params = calibrator.get_calibration_params()
    print("Calibration parameters:", params)
    
    # Get validation metrics
    metrics = calibrator.get_validation_metrics()
    print("Validation metrics:", metrics)
    
    # Apply to new data
    new_mag_data = [
        {'x': 125, 'y': 458, 'z': 787, 'timestamp': 1609459202},
        # More new magnetometer data...
    ]
    
    new_rpm_data = [
        {'rpm_1': 60, 'rpm_2': 40, 'rpm_3': 60, 'rpm_4': 40, 'timestamp': 1609459202},
        # More new RPM data...
    ]
    
    new_headings = calibrator.apply_calibration_to_new_data(new_mag_data, new_rpm_data)
    
    return calibrated_headings, new_headings

if __name__ == "__main__":
    example()