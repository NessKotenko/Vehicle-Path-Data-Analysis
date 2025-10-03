#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Set print options to suppress scientific notation and set a high precision
np.set_printoptions(suppress=True, precision=5)

data = np.load('vehicle_path.npz')

print(data)

loaded_array1 = data['path']

print(loaded_array1)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

data = np.load('vehicle_path.npz')
loaded_array = data['path']

x_coords = loaded_array[:, 0]
y_coords = loaded_array[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')
plt.title('Vehicle Path')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.tight_layout()

plt.savefig('vehicle_path.png')

plt.show()


# In[3]:


data = np.load('vehicle_path.npz')
loaded_array = data['path']
print(loaded_array)


# In[4]:


#Cleaning the Data:

import numpy as np

def remove_nan_inf_rows(arr):

    # Check for NaN and infinite values across the entire array
    valid_rows = ~np.any(np.isnan(arr) | np.isinf(arr), axis=1)
    
    # Return the array with only the valid rows
    return arr[valid_rows]

# Example usage:
data = np.load('vehicle_path.npz')
loaded_vehicle_path_array = data['path']
cleaned_vehicle_path_array = remove_nan_inf_rows(loaded_vehicle_path_array)
print(cleaned_vehicle_path_array)


# In[5]:


x_coords = cleaned_vehicle_path_array[:, 0]
y_coords = cleaned_vehicle_path_array[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')
plt.title('Vehicle Path')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.tight_layout()

plt.savefig('vehicle_path.png')

plt.show()


# In[6]:


import numpy as np

data2 = np.load('traffic_lights.npz')


print(data2)

loaded_traffic_lights_positions_array = data2['positions']

print(loaded_traffic_lights_positions_array)


# In[7]:


x_coords = loaded_traffic_lights_positions_array[:, 0]
y_coords = loaded_traffic_lights_positions_array[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(x_coords, y_coords, marker='o', color='b')
plt.title('Traffic Lights')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.tight_layout()

plt.savefig('traffic_lights.png')

plt.show()


# In[8]:


#GRAPH PLOTS WITH BOTH VISUALS (Vehicle path & Traffic Lights)

# Load vehicle path data
vehicle_path_x_coords = cleaned_vehicle_path_array[:, 0]
vehicle_path_y_coords = cleaned_vehicle_path_array[:, 1]

# Load traffic lights data
traffic_lights_x_coords = loaded_traffic_lights_positions_array[:, 0]
traffic_lights_y_coords = loaded_traffic_lights_positions_array[:, 1]

# Create a single plot
plt.figure(figsize=(8,6))

# Plot the vehicle path as a line plot
plt.plot(vehicle_path_x_coords, vehicle_path_y_coords,
         label='Vehicle Path', color='blue', marker='o', linestyle='-')

# Plot the traffic lights as a scatter plot
plt.scatter(traffic_lights_x_coords, traffic_lights_y_coords,
            label='Traffic Lights', color='red', marker='X', s=100)

# Add title, labels, and legend
plt.title('Vehicle Path and Traffic Lights')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the merged plot
plt.savefig('merged_plot.png')


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


loaded_array = cleaned_vehicle_path_array

# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(loaded_array, columns=['x_coords', 'y_coords'])

# --- Smoothing using pandas DataFrame columns ---

# Moving Average Smoothing
# Use the .rolling() method for a clean moving average calculation
window_size = 10
df['x_smooth_ma'] = df['x_coords'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_coords'].rolling(window=window_size).mean()


# --- Plotting the original and smoothed paths ---
plt.figure(figsize=(10, 8))

# Plot the original path directly from the DataFrame
plt.plot(df['x_coords'], df['y_coords'], 'o-', label='Original Path', color='gray', alpha=0.5)

# Plot the moving average smoothed path from the DataFrame
plt.plot(df['x_smooth_ma'], df['y_smooth_ma'], 's--', label='Moving Average', color='purple')

# Add labels and legend
plt.title('Vehicle Path Smoothing (Moving Avarage)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('smoothed_path_ma.png')


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


loaded_array = cleaned_vehicle_path_array

# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(loaded_array, columns=['x_coords', 'y_coords'])

# --- Smoothing using pandas DataFrame columns ---

# Moving Average Smoothing
# Use the .rolling() method for a clean moving average calculation
window_size = 5
df['x_smooth_ma'] = df['x_coords'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_coords'].rolling(window=window_size).mean()

df['x_smooth_ma'] = df['x_smooth_ma'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_smooth_ma'].rolling(window=window_size).mean()

df['x_smooth_ma'] = df['x_smooth_ma'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_smooth_ma'].rolling(window=window_size).mean()


# --- Plotting the original and smoothed paths ---
plt.figure(figsize=(10, 8))

# Plot the original path directly from the DataFrame
plt.plot(df['x_coords'], df['y_coords'], 'o-', label='Original Path', color='gray', alpha=0.5)

# Plot the moving average smoothed path from the DataFrame
plt.plot(df['x_smooth_ma'], df['y_smooth_ma'], 's--', label='Moving Average', color='purple')

# Add labels and legend
plt.title('Vehicle Path Smoothing (Moving Avarage)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('smoothed_path_ma_three_times.png')


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


loaded_array = cleaned_vehicle_path_array

# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(loaded_array, columns=['x_coords', 'y_coords'])

# --- Smoothing using pandas DataFrame columns ---

# Spline Interpolation
# Extract the x and y data from the DataFrame for the interpolation function
x_coords = df['x_coords']
y_coords = df['y_coords']
x_new = np.linspace(x_coords.min(), x_coords.max(), 300)
f_x = interp1d(x_coords, y_coords, kind='cubic')
y_smooth_spline = f_x(x_new)

# --- Plotting the original and smoothed paths ---
plt.figure(figsize=(10, 8))

# Plot the original path directly from the DataFrame
plt.plot(df['x_coords'], df['y_coords'], 'o-', label='Original Path', color='gray', alpha=0.5)

# Plot the spline interpolation smoothed path
plt.plot(x_new, y_smooth_spline, '-', label='Spline Interpolation', color='green')

# Add labels and legend
plt.title('Vehicle Path Smoothing (Spline Interpolation)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('smoothed_path_spline.png')


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


loaded_array = cleaned_vehicle_path_array

# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(loaded_array, columns=['x_coords', 'y_coords'])

# --- Smoothing using pandas DataFrame columns ---

# Moving Average Smoothing
# Use the .rolling() method for a clean moving average calculation
window_size = 5
df['x_smooth_ma'] = df['x_coords'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_coords'].rolling(window=window_size).mean()

df['x_smooth_ma'] = df['x_smooth_ma'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_smooth_ma'].rolling(window=window_size).mean()

df['x_smooth_ma'] = df['x_smooth_ma'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_smooth_ma'].rolling(window=window_size).mean()

# Spline Interpolation
# Extract the x and y data from the DataFrame for the interpolation function
x_coords = df['x_coords']
y_coords = df['y_coords']
x_new = np.linspace(x_coords.min(), x_coords.max(), 300)
f_x = interp1d(x_coords, y_coords, kind='cubic')
y_smooth_spline = f_x(x_new)

# --- Plotting the original and smoothed paths ---
plt.figure(figsize=(10, 8))
plt.gca().set_aspect('equal', adjustable='box') 

# Plot the original path directly from the DataFrame
plt.plot(df['x_coords'], df['y_coords'], 'o-', label='Original Path', color='gray', alpha=0.5)

# Plot the moving average smoothed path from the DataFrame
plt.plot(df['x_smooth_ma'], df['y_smooth_ma'], 's--', label='Moving Average', color='purple')

# Plot the spline interpolation smoothed path
plt.plot(x_new, y_smooth_spline, '-', label='Spline Interpolation', color='green')

# Add labels and legend
plt.title('Vehicle Path Smoothing (from DataFrame)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('smoothed_path_df.png')


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

loaded_array = cleaned_vehicle_path_array

# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(loaded_array, columns=['x_coords', 'y_coords'])

# --- Moving Average Smoothing ---
window_size = 5
df['x_smooth_ma'] = df['x_coords'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_coords'].rolling(window=window_size).mean()

df['x_smooth_ma'] = df['x_smooth_ma'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_smooth_ma'].rolling(window=window_size).mean()

df['x_smooth_ma'] = df['x_smooth_ma'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_smooth_ma'].rolling(window=window_size).mean()

# --- Spline Interpolation based on Moving Average Points ---
# Clean the moving average data to remove NaNs
df_ma_cleaned = df.dropna()

x_ma_cleaned = df_ma_cleaned['x_smooth_ma']
y_ma_cleaned = df_ma_cleaned['y_smooth_ma']

# Use the cleaned moving average points for interpolation
f_x = interp1d(x_ma_cleaned, y_ma_cleaned, kind='cubic')
x_new = np.linspace(x_ma_cleaned.min(), x_ma_cleaned.max(), 300)
y_smooth_spline = f_x(x_new)

# --- Plotting the original and smoothed paths ---
plt.figure(figsize=(10, 8))

# Plot the original path
plt.plot(df['x_coords'], df['y_coords'], 'o-', label='Original Path', color='gray', alpha=0.5)

# Plot the moving average smoothed path
plt.plot(df['x_smooth_ma'], df['y_smooth_ma'], 's--', label='Moving Average', color='purple')

# Plot the spline interpolation smoothed path
plt.plot(x_new, y_smooth_spline, '-', label='Spline Interpolation (on MA)', color='green')

# Add labels and legend
plt.title('Vehicle Path Smoothing (MA-based Spline)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('smoothed_path_ma_spline.png')


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

loaded_array = cleaned_vehicle_path_array

# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(loaded_array, columns=['x_coords', 'y_coords'])

# --- Moving Average Smoothing ---
window_size = 5
df['x_smooth_ma'] = df['x_coords'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_coords'].rolling(window=window_size).mean()

df['x_smooth_ma'] = df['x_smooth_ma'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_smooth_ma'].rolling(window=window_size).mean()

df['x_smooth_ma'] = df['x_smooth_ma'].rolling(window=window_size).mean()
df['y_smooth_ma'] = df['y_smooth_ma'].rolling(window=window_size).mean()

# --- Spline Interpolation based on Moving Average Points ---
# Clean the moving average data to remove NaNs
df_ma_cleaned = df.dropna()

x_ma_cleaned = df_ma_cleaned['x_smooth_ma']
y_ma_cleaned = df_ma_cleaned['y_smooth_ma']

# Use the cleaned moving average points for interpolation
f_x = interp1d(x_ma_cleaned, y_ma_cleaned, kind='cubic')
x_new = np.linspace(x_ma_cleaned.min(), x_ma_cleaned.max(), 3000)
y_smooth_spline = f_x(x_new)

# --- Plotting the original and smoothed paths ---
plt.figure(figsize=(10, 8))
plt.gca().set_aspect('equal', adjustable='box') 

# Plot the original path
plt.plot(df['x_coords'], df['y_coords'], 'o-', label='Original Path', color='gray', alpha=0.5)

# Plot the moving average smoothed path
plt.plot(df['x_smooth_ma'], df['y_smooth_ma'], 's--', label='Moving Average', color='purple')

# Plot the spline interpolation smoothed path
plt.plot(x_new, y_smooth_spline, '-', label='Spline Interpolation (on MA)', color='green')

# Load traffic lights data
traffic_lights_x_coords = loaded_traffic_lights_positions_array[:, 0]
traffic_lights_y_coords = loaded_traffic_lights_positions_array[:, 1]
# Plot the traffic lights as a scatter plot
plt.scatter(traffic_lights_x_coords, traffic_lights_y_coords,
            label='Traffic Lights', color='red', marker='X', s=100)

# Add labels and legend
plt.title('Vehicle Path Smoothing (MA-based Spline)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('smoothed_path_ma_spline.png')


# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

loaded_array = cleaned_vehicle_path_array
vehicle_path_df = pd.DataFrame(loaded_array, columns=['x_coords', 'y_coords'])
df = vehicle_path_df
traffic_lights = loaded_traffic_lights_positions_array

# --- Corrected closest point function ---
def find_closest_point(vehicle_path, traffic_lights):
  
    results = []
    # Convert path to NumPy array for faster calculations
    path_coords = vehicle_path[['x_coords', 'y_coords']].values

    for i, light in enumerate(traffic_lights):
        # Calculate the Euclidean distance from the traffic light to every point on the path
        distances = np.linalg.norm(path_coords - light, axis=1)
        
        # Find the index of the minimum distance
        min_dist_index = np.argmin(distances)

        # Get the closest point and its distance
        min_dist = distances[min_dist_index]
        closest_point = path_coords[min_dist_index]

        results.append({
            'light_x': light[0],
            'light_y': light[1],
            'closest_x': closest_point[0],
            'closest_y': closest_point[1],
            'distance': min_dist
        })
    return pd.DataFrame(results)

# --- Perform analysis and get results ---
closest_points_df = find_closest_point(df, traffic_lights)
print(closest_points_df)

# --- Plot the original path and the traffic light analysis ---
plt.figure(figsize=(10, 8))
plt.gca().set_aspect('equal', adjustable='box') 

plt.plot(df['x_coords'], df['y_coords'], 'o-', label='Original Path', color='gray', alpha=0.5)
plt.plot(traffic_lights[:, 0], traffic_lights[:, 1], 'ro', markersize=10, label='Traffic Lights')
plt.plot(closest_points_df['closest_x'], closest_points_df['closest_y'], 'gs', markersize=10, label='Closest Points on Path')

# Draw lines connecting each traffic light to its closest point
for i, row in closest_points_df.iterrows():
    plt.plot([row['light_x'], row['closest_x']], [row['light_y'], row['closest_y']], 'k--')

plt.title('Closest Point on Path to Each Traffic Light (Discrete Points)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('traffic_light_analysis.png')


# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def remove_nan_inf_rows(arr):
    # Check for NaN and infinite values across the entire array
    valid_rows = ~np.any(np.isnan(arr) | np.isinf(arr), axis=1)
    # Return the array with only the valid rows
    return arr[valid_rows]


def find_closest_bruteforce(smoothed_path_df, traffic_lights_array):

    results = []
    
    # Convert path to NumPy array for faster vectorized distance calculations
    smoothed_points = smoothed_path_df[['x_smooth_ma', 'y_smooth_ma']].values

    for light in traffic_lights_array:
        
        # Calculate the Euclidean distance from the traffic light (light) to 
        # EVERY point (smoothed_points) on the path using vectorization.
        # axis=1 ensures distance is calculated for each row (point).
        distances = np.linalg.norm(smoothed_points - light, axis=1)
        
        # Find the index of the minimum distance (the closest path point)
        min_dist_index = np.argmin(distances)

        # Get the corresponding closest point and the distance
        min_dist = distances[min_dist_index]
        closest_point = smoothed_points[min_dist_index]

        results.append({
            'light_x': light[0],
            'light_y': light[1],
            'closest_x': closest_point[0],
            'closest_y': closest_point[1],
            'distance': min_dist
        })
    
    # Return results as a DataFrame
    return pd.DataFrame(results)

def load_and_clean(vehicle_path_filename = 'vehicle_path.npz', traffic_lights_filename = 'traffic_lights.npz'):
        
    # Load Vehicle Path Data from file
    loaded_vehicle_path_data = np.load(vehicle_path_filename)
    cleaned_vehicle_path_array = remove_nan_inf_rows(loaded_vehicle_path_data['path'])
    
    # Convert the NumPy array to a pandas DataFrame
    vehicle_path_df = pd.DataFrame(cleaned_vehicle_path_array, columns=['x_coords', 'y_coords'])

    # --- Load Traffic Light Points ---
    traffic_lights_data = np.load(traffic_lights_filename)
    traffic_lights_array = traffic_lights_data['positions']

    return vehicle_path_df , traffic_lights_array

def smooth_path_moving_avg(vehicle_path_df, window_size = 5):
    
    vehicle_path_df['x_smooth_ma'] = vehicle_path_df['x_coords'].rolling(window=window_size).mean()
    vehicle_path_df['y_smooth_ma'] = vehicle_path_df['y_coords'].rolling(window=window_size).mean()
    smoothed_vehicle_path_df = vehicle_path_df.dropna()

    return smoothed_vehicle_path_df


def run_brute_force_alg(smoothed_vehicle_path_df, traffic_lights):
    
    
    # --- Perform Analysis and Get Results ---
    closest_points_results_brute_force_df = find_closest_bruteforce(smoothed_vehicle_path_df, traffic_lights)
    
    return closest_points_results_brute_force_df

    
# --- Load, Clean & Create a new smoothed path with equally spaced points ---
    
vehicle_path_df , traffic_lights_array = load_and_clean()

smoothed_vehicle_path_df = smooth_path_moving_avg(vehicle_path_df, window_size = 10)

#Find closest points with brute force algorithm
closest_points_results_brute_force_df = run_brute_force_alg(smoothed_vehicle_path_df, traffic_lights_array)



# Print results to confirm the closest points
print("Closest points found using Brute-Force:")
print(closest_points_results_brute_force_df)

# --- Plot the results ---
plt.figure(figsize=(10, 8))
plt.plot(smoothed_vehicle_path_df['x_smooth_ma'], smoothed_vehicle_path_df['y_smooth_ma'], 'o-', 
         label='Smoothed Path (Moving Average)', color='orange', alpha=0.5)

plt.plot(traffic_lights_array[:, 0], traffic_lights_array[:, 1], 'ro', 
         markersize=10, label='Traffic Lights')

plt.plot(closest_points_results_brute_force_df['closest_x'], closest_points_results_brute_force_df['closest_y'], 
         'gs', markersize=10, label='Closest Points on Smoothed Path')

# Draw dashed lines connecting each traffic light to its closest point
for i, row in closest_points_results_brute_force_df.iterrows():
    plt.plot([row['light_x'], row['closest_x']], [row['light_y'], row['closest_y']], 'k--')

plt.gca().set_aspect('equal', adjustable='box') 
plt.title('Closest Points on Smoothed Path (Brute-Force Algorithm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('closest_points_brute_force.png')


# In[143]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
#from sklearn.neighbors import KDTree


def find_closest_ann_kdtree(smoothed_vehicle_path_df, traffic_lights_array):

    # 1. Preprocessing: Build the K-D tree from the smoothed path points
    smoothed_points = smoothed_vehicle_path_df[['x_smooth_ma', 'y_smooth_ma']].values
    tree = cKDTree(smoothed_points, leafsize=3)
    #tree = KDTree(smoothed_points, leaf_size=3)

    # 2. Querying: Find the nearest neighbor for each traffic light
    distances, indices = tree.query(traffic_lights_array, k=1)

    #print(distances)
    #min_dist_index = np.argmin(distances)
    #min_dist = distances[min_dist_index]
    
    results = []
    for i, light in enumerate(traffic_lights_array):
        closest_point = smoothed_points[indices[i]]
        results.append({
            'light_x': light[0],
            'light_y': light[1],
            'closest_x': closest_point[0],
            'closest_y': closest_point[1],
            'distance': distances[i]
        })
    
    return pd.DataFrame(results)

    
def run_ann_kdtree_alg(smoothed_vehicle_path_df , traffic_lights_array):
    

    # --- Find the closest points using the ANN function ---
    closest_points_results_ann_kdtree_df = find_closest_ann_kdtree(smoothed_vehicle_path_df, traffic_lights_array)
    
    return closest_points_results_ann_kdtree_df


# --- Load, Clean & Create a new smoothed path with equally spaced points ---
    
vehicle_path_df , traffic_lights_array = load_and_clean()

smoothed_vehicle_path_df = smooth_path_moving_avg(vehicle_path_df, window_size = 10)

#Find closest points with ANN KDTree algorithm
closest_points_results_ann_kdtree_df = run_ann_kdtree_alg(smoothed_vehicle_path_df, traffic_lights_array)



# Print results
print("Closest points found using ANN KDTree Algorithm:")
print(closest_points_results_ann_kdtree_df)
    
# --- Plot the results ---
plt.figure(figsize=(10, 8))
plt.plot(smoothed_vehicle_path_df['x_smooth_ma'], smoothed_vehicle_path_df['y_smooth_ma'], 'o-', 
         label='Smoothed Path (Moving Average)', color='orange', alpha=0.5)

plt.plot(traffic_lights_array[:, 0], traffic_lights_array[:, 1], 'ro', 
         markersize=10, label='Traffic Lights')

plt.plot(closest_points_results_ann_kdtree_df['closest_x'], closest_points_results_ann_kdtree_df['closest_y'], 
         'gs', markersize=10, label='Closest Points on Smoothed Path')

# Draw dashed lines connecting each traffic light to its closest point
for i, row in closest_points_results_ann_kdtree_df.iterrows():
    plt.plot([row['light_x'], row['closest_x']], [row['light_y'], row['closest_y']], 'k--')

plt.gca().set_aspect('equal', adjustable='box') 
plt.title('Closest Points on Smoothed Path (ANN KDTree Algorithm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('closest_points_ann_kdtree.png')


# In[105]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree


def find_closest_ann_balltree(smoothed_path, traffic_lights):

    smoothed_points = smoothed_path[['x_smooth_ma', 'y_smooth_ma']].values
   
    path_tree = BallTree(smoothed_points)
    distances, indices = path_tree.query(traffic_lights, k=1)
    # Flatten the results for easier use
    distances = distances.flatten()
    indices = indices.flatten()
 
    results = []
    for i, light in enumerate(traffic_lights):
        closest_point = smoothed_points[indices[i]]
        results.append({
            'light_x': light[0],
            'light_y': light[1],
            'closest_x': closest_point[0],
            'closest_y': closest_point[1]
        })
    
    return pd.DataFrame(results)


def run_ann_balltree_alg(smoothed_vehicle_path_df , traffic_lights_array):
    

    # --- Find the closest points using the ANN function ---
    closest_points_ann_balltree_df = find_closest_ann_balltree(smoothed_vehicle_path_df, traffic_lights_array)
    
    return closest_points_ann_balltree_df



# --- Load, Clean & Create a new smoothed path with equally spaced points ---   
vehicle_path_df , traffic_lights_array = load_and_clean()
smoothed_vehicle_path_df = smooth_path_moving_avg(vehicle_path_df, window_size = 10)

#Find closest points with ANN BallTree algorithm
closest_points_results_ann_balltree_df = run_ann_balltree_alg(smoothed_vehicle_path_df, traffic_lights_array)



# Print results
print("Closest points found using ANN BallTree Algorithm")
print(closest_points_results_ann_balltree_df)


# --- Plot the results ---
plt.figure(figsize=(10, 8))
plt.plot(smoothed_vehicle_path_df['x_smooth_ma'], smoothed_vehicle_path_df['y_smooth_ma'], 'o-', 
         label='Smoothed Path (Moving Average)', color='orange', alpha=0.5)

plt.plot(traffic_lights_array[:, 0], traffic_lights_array[:, 1], 'ro', 
         markersize=10, label='Traffic Lights')

plt.plot(closest_points_results_ann_balltree_df['closest_x'], closest_points_results_ann_balltree_df['closest_y'], 
         'gs', markersize=10, label='Closest Points on Smoothed Path')

# Draw dashed lines connecting each traffic light to its closest point
for i, row in closest_points_results_ann_balltree_df.iterrows():
    plt.plot([row['light_x'], row['closest_x']], [row['light_y'], row['closest_y']], 'k--')

plt.gca().set_aspect('equal', adjustable='box') 
plt.title('Closest Points on Smoothed Path (ANN BallTree Algorithm)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('closest_points_ann_balltree.png')


# In[115]:


import numpy as np
import pandas as pd

def calculate_path_length(smoothed_vehicle_path_df, x_index = 'x_smooth_ma', y_index = 'y_smooth_ma'):
    
    # 1. Extract coordinates into a NumPy array for efficiency
    
    points = smoothed_vehicle_path_df[[x_index, y_index]].values
    
    # 2. Calculate the difference between consecutive points (Delta X, Delta Y)
    deltas = points[1:] - points[:-1]
    
    # 3. Calculate the Euclidean distance (norm) for each segment
    # np.linalg.norm([dx, dy]) calculates sqrt(dx^2 + dy^2) for each segment.
    segment_lengths = np.linalg.norm(deltas, axis=1)
    
    # 4. Sum the lengths of all segments to get the total path length
    total_length = np.sum(segment_lengths)
    
    return total_length


# --- Load, Clean & Create a new smoothed path ---   
vehicle_path_df , traffic_lights_array = load_and_clean()
smoothed_vehicle_path_df = smooth_path_moving_avg(vehicle_path_df, window_size = 10)

# Calculate the length
path_length = calculate_path_length(smoothed_vehicle_path_df)

print(f"The total length of the smoothed vehicle path is: {path_length:.4f} units")


# In[117]:


import numpy as np
import pandas as pd

def calculate_avg_segment_distance(smoothed_vehicle_path_df):
    
    
    total_length = calculate_path_length(smoothed_vehicle_path_df)

    num_segments = len(smoothed_vehicle_path_df)
    
    if num_segments == 0:
        return 0.0 # Handle case with 0 or 1 point
    
    average_distance = total_length / num_segments
    
    return average_distance

# --- Load, Clean & Create a new smoothed path ---   
vehicle_path_df , traffic_lights_array = load_and_clean()
smoothed_vehicle_path_df = smooth_path_moving_avg(vehicle_path_df, window_size = 10)

# Calculate the average distance
avg_segment_distance = calculate_avg_segment_distance(smoothed_vehicle_path_df)

print(f"The average distance between consecutive points of the smoothed path is: {avg_segment_distance:.6f} units")


# In[490]:


import numpy as np
import pandas as pd

def calculate_avg_segment_distance(vehicle_path_df, x_index = "x_smooth_ma", y_index = 'y_smooth_ma'):
    
    
    total_length = calculate_path_length(vehicle_path_df, x_index , y_index)

    num_segments = len(vehicle_path_df)
    
    if num_segments == 0:
        return 0.0 # Handle case with 0 or 1 point
    
    average_distance = total_length / num_segments
    
    return average_distance


# --- Load & Clean original path data ---   
vehicle_path_df , traffic_lights_array = load_and_clean()

# Calculate the average distance on original data points
avg_segment_distance = calculate_avg_segment_distance(vehicle_path_df, 'x_coords', 'y_coords')


print(f"The average distance between consecutive points of the original path data is: {avg_segment_distance:.6f} units")


# In[145]:


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree



def search_performance_comparison_test(vehicle_path_df, traffic_lights_arr, subset_size_incrementor = 10):

    full_path_size = len(vehicle_path_df)
    
    # --- 3. Define the data subsets for the test ---
    # Create subsets of the data in steps of 10 points
    test_sizes = list(range(10, full_path_size + 1, subset_size_incrementor))
    if full_path_size % 10 != 0:
        test_sizes.append(full_path_size)
    test_sizes = sorted(list(set(test_sizes))) # Remove duplicates and sort
    
    brute_force_runtimes = []
    ann_kdtree_runtimes = []
    ann_balltree_runtimes = []
    
    # --- 4. Run the performance test loop ---
    for size in test_sizes:
        # Get the current data subset
        subset_df = vehicle_path_df.iloc[:size]
    
        # Measure Brute-Force runtime
        start_time = time.time()
        closest_points_results_brute_force_df = run_brute_force_alg(subset_df, traffic_lights_arr)
        brute_force_runtimes.append(time.time() - start_time)
#        print(f"Brute-Force Results for dataset size {size} : \n")
#        print(closest_points_results_brute_force_df)
    
        # Measure ANN KDTree runtime
        start_time = time.time()
        closest_points_results_ann_kdtree_df = run_ann_kdtree_alg(subset_df, traffic_lights_arr)
        ann_kdtree_runtimes.append(time.time() - start_time)
#        print(f"ANN KDTree Results for dataset size {size} : \n")
#        print(closest_points_results_ann_kdtree_df)

        # Measure ANN BallTree runtime
        start_time = time.time()
        closest_points_results_ann_balltree_df = run_ann_balltree_alg(subset_df, traffic_lights_arr)
        ann_balltree_runtimes.append(time.time() - start_time)
#        print(f"ANN BallTree Results for dataset size {size} : \n")
#        print(closest_points_results_ann_balltree_df)
        
#        print(f"Tested with {size} points.")
    
    return test_sizes, brute_force_runtimes, ann_kdtree_runtimes, ann_balltree_runtimes


#--Load & Clean datasets---
vehicle_path_df, traffic_lights_array = load_and_clean()
smoothed_vehicle_path_df = smooth_path_moving_avg(vehicle_path_df, window_size = 10)

#--- Run Performance Comparison Test ---
test_sizes, brute_force_runtimes, ann_kdtree_runtimes, ann_balltree_runtimes = search_performance_comparison_test(smoothed_vehicle_path_df, traffic_lights_array)

# --- 5. Plot the results to visually compare performance ---
plt.figure(figsize=(10, 6))
plt.plot(test_sizes, brute_force_runtimes, 'o-', label='Brute-Force', color='red')
plt.plot(test_sizes, ann_kdtree_runtimes, 's-', label='ANN (cKDTree)', color='blue')
plt.plot(test_sizes, ann_balltree_runtimes, '^-', label='ANN (BallTree)', color='green')

plt.title('Algorithm Runtime Comparison on Real Data')
plt.xlabel('Number of Path Points')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.tight_layout()

plt.grid(True)
plt.show()

print("\nPerformance test complete.")


# In[121]:


import numpy as np
import pandas as pd

def create_synthetic_path_and_traffic_lights(
    path_points_count: int = 10000, 
    traffic_lights_count: int = 50,
    amplitude: float = 1.0, 
    frequency: float = 0.5, 
    noise_level: float = 0.2
):
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
    np.savez_compressed('vehicle_path_synthetic.npz', path=synthetic_path)
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
    np.savez_compressed('traffic_lights_synthetic.npz', positions=synthetic_traffic_lights)
    print(f'Generated {traffic_lights_count} traffic light positions and saved to traffic_lights_synthetic.npz')


##CREATE SYNTH DATA
#create_synthetic_path_and_traffic_lights(path_points_count=10000, traffic_lights_count=100)


# In[123]:


##CREATE SYNTH DATA
create_synthetic_path_and_traffic_lights(path_points_count=2000, traffic_lights_count=100)


#LOAD SYNTH DATA
vehicle_path_filename = 'vehicle_path_synthetic.npz'
traffic_lights_filename = 'traffic_lights_synthetic.npz'

vehicle_path_df , traffic_lights_array = load_and_clean(vehicle_path_filename, traffic_lights_filename)



# Moving Average Smoothing
# Use the .rolling() method for a clean moving average calculation
window_size = 20
vehicle_path_df['x_smooth_ma'] = vehicle_path_df['x_coords'].rolling(window=window_size).mean()
vehicle_path_df['y_smooth_ma'] = vehicle_path_df['y_coords'].rolling(window=window_size).mean()
vehicle_path_df = vehicle_path_df.dropna()


## Spline Interpolation
## Extract the x and y data from the DataFrame for the interpolation function
#x_coords = vehicle_path_df['x_coords']
#y_coords = vehicle_path_df['y_coords']
#x_new = np.linspace(x_coords.min(), x_coords.max(), 5000)
#f_x = interp1d(x_coords, y_coords, kind='cubic')
#y_smooth_spline = f_x(x_new)



#VIEW DATA

# --- Plotting the original and smoothed paths ---
plt.figure(figsize=(200, 20))
plt.gca().set_aspect('equal', adjustable='box') 

# Plot the original path directly from the DataFrame
plt.plot(vehicle_path_df['x_coords'], vehicle_path_df['y_coords'], 'o-', label='Original Path', color='gray', alpha=0.5)

# Plot the moving average smoothed path from the DataFrame
plt.plot(vehicle_path_df['x_smooth_ma'], vehicle_path_df['y_smooth_ma'], 's--', label='Moving Average', color='purple')

# Plot the spline interpolation smoothed path
#plt.plot(x_new, y_smooth_spline, '-', label='Spline Interpolation', color='green')

plt.scatter(traffic_lights_array[:, 0], traffic_lights_array[:, 1], label='Traffic Lights', color='red', marker='X', s=100)


# Add labels and legend
plt.title('Vehicle Path Smoothing (from DataFrame)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()
#plt.tight_layout()
plt.savefig('synthetic_data_path_trafficlights_smoothed.png')


# In[147]:


##CREATE SYNTH DATA
create_synthetic_path_and_traffic_lights(path_points_count=10000, traffic_lights_count=300)


#LOAD SYNTH DATA
vehicle_path_filename = 'vehicle_path_synthetic.npz'
traffic_lights_filename = 'traffic_lights_synthetic.npz'

vehicle_path_df , traffic_lights_array = load_and_clean(vehicle_path_filename, traffic_lights_filename)



# Moving Average Smoothing
# Use the .rolling() method for a clean moving average calculation
window_size = 20
vehicle_path_df['x_smooth_ma'] = vehicle_path_df['x_coords'].rolling(window=window_size).mean()
vehicle_path_df['y_smooth_ma'] = vehicle_path_df['y_coords'].rolling(window=window_size).mean()
smoothed_vehicle_path_df = vehicle_path_df.dropna()


#--- Run Performance Comparison Test ---
test_sizes, brute_force_runtimes, ann_kdtree_runtimes, ann_balltree_runtimes = search_performance_comparison_test(smoothed_vehicle_path_df, traffic_lights_array,100)

# --- 5. Plot the results to visually compare performance ---
plt.figure(figsize=(10, 6))
plt.plot(test_sizes, brute_force_runtimes, 'o-', label='Brute-Force', color='red')
plt.plot(test_sizes, ann_kdtree_runtimes, 's-', label='ANN (cKDTree)', color='blue')
plt.plot(test_sizes, ann_balltree_runtimes, '^-', label='ANN (BallTree)', color='green')

plt.title('Algorithm Runtime Comparison on Real Data')
plt.xlabel('Number of Path Points')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.tight_layout()

plt.grid(True)
plt.show()

print("\nPerformance test complete.")

