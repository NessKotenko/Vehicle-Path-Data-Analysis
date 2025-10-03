import os


def remove_nan_inf_rows(arr):
    
    import numpy as np

    # Check for NaN and infinite values across the entire array
    valid_rows = ~np.any(np.isnan(arr) | np.isinf(arr), axis=1)
    
    # Return the array with only the valid rows
    return arr[valid_rows]

vehicle_path_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'Data Files', 'Original', 'vehicle_path.npz')
traffic_lights_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'Data Files', 'Original', 'traffic_lights.npz')


#Load and Clean data
def load_and_clean(vehicle_path_filename = vehicle_path_file_path, traffic_lights_filename = traffic_lights_file_path):
      
    import numpy as np
    import pandas as pd

    # Load Vehicle Path Data from file

    loaded_vehicle_path_data = np.load(vehicle_path_filename)
    cleaned_vehicle_path_array = remove_nan_inf_rows(loaded_vehicle_path_data['path'])
    
    # Convert the NumPy array to a pandas DataFrame
    vehicle_path_df = pd.DataFrame(cleaned_vehicle_path_array, columns=['x_coords', 'y_coords'])

    # --- Load Traffic Light Points ---
    traffic_lights_data = np.load(traffic_lights_filename)
    traffic_lights_array = traffic_lights_data['positions']

    return vehicle_path_df , traffic_lights_array




def create_synthetic_path_and_traffic_lights(
    path_points_count: int = 10000, 
    traffic_lights_count: int = 50,
    amplitude: float = 1.0, 
    frequency: float = 0.5, 
    noise_level: float = 0.2
):
    import numpy as np

    """
    Generates and saves synthetic vehicle path and traffic light data.

    Args:
        path_points_count (int): The total number of points in the vehicle path.
        traffic_lights_count (int): The number of traffic light points to generate.
        amplitude (float): The amplitude of the underlying sine wave.
        frequency (float): The frequency of the underlying sine wave.
        noise_level (float): The magnitude of the random noise added to the path.
    """
    # --- Generate Vehicle Path Data ---
    x_coords = np.linspace(0, 200, path_points_count)
    
    # Generate a sine wave with specified amplitude and frequency
    y_coords_base = amplitude * np.sin(frequency * x_coords)
    
    # Add random noise to the sine wave
    noise = np.random.normal(0, noise_level, path_points_count)
    y_coords_noisy = y_coords_base + noise
    
    synthetic_path = np.vstack([x_coords, y_coords_noisy]).T
    
    # Save the synthetic path data

    synth_vehicle_path_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'Data Files', 'Synthetic', 'vehicle_path_synthetic.npz')

    np.savez_compressed(synth_vehicle_path_file_path, path=synthetic_path)
    print(f'Generated {path_points_count} path points and saved to vehicle_path_synthetic.npz')
    
    # --- Generate Traffic Lights Data ---
    # Traffic lights are scattered near the path with some noise
    
    # Use a subset of the path points to place traffic lights
    indices = np.random.choice(range(path_points_count), traffic_lights_count, replace=False)
    
    # Add a bit more random offset for a more realistic spread
    traffic_x = x_coords[indices] + np.random.uniform(-1, 1, traffic_lights_count)
    traffic_y = y_coords_noisy[indices] + np.random.uniform(-1, 1, traffic_lights_count)
    
    synthetic_traffic_lights = np.vstack([traffic_x, traffic_y]).T
    
    # Save the synthetic traffic lights data
    synth_traffic_lights_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'Data Files', 'Synthetic', 'traffic_lights_synthetic.npz')

    np.savez_compressed(synth_traffic_lights_file_path, positions=synthetic_traffic_lights)
    print(f'Generated {traffic_lights_count} traffic light positions and saved to traffic_lights_synthetic.npz')

    return "Successfully created synthetic dataset file"




