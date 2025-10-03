

def run_brute_force_alg(smoothed_vehicle_path_df, traffic_lights):
    # --- Perform Analysis and Get Results ---
    closest_points_results_brute_force_df = find_closest_bruteforce(smoothed_vehicle_path_df, traffic_lights)
    return closest_points_results_brute_force_df


def run_ann_kdtree_alg(smoothed_vehicle_path_df , traffic_lights_array):
    # --- Find the closest points using the ANN function ---
    closest_points_results_ann_kdtree_df = find_closest_ann_kdtree(smoothed_vehicle_path_df, traffic_lights_array)
    return closest_points_results_ann_kdtree_df

def run_ann_balltree_alg(smoothed_vehicle_path_df , traffic_lights_array):
    # --- Find the closest points using the ANN function ---
    closest_points_ann_balltree_df = find_closest_ann_balltree(smoothed_vehicle_path_df, traffic_lights_array) 
    return closest_points_ann_balltree_df


def find_closest_bruteforce(smoothed_path_df, traffic_lights_array):
    
    import pandas as pd
    import numpy as np
    
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


def find_closest_ann_kdtree(smoothed_vehicle_path_df, traffic_lights_array):

    from sklearn.neighbors import KDTree
    import pandas as pd
   
    #Preprocessing: Build the K-D tree from the smoothed path points
    smoothed_points = smoothed_vehicle_path_df[['x_smooth_ma', 'y_smooth_ma']].values
    tree = KDTree(smoothed_points, leaf_size=3)
    
    #Querying: Find the nearest neighbor for each traffic light
    distances, indices = tree.query(traffic_lights_array, k=1)

    results = []
    for i, light in enumerate(traffic_lights_array):
        closest_point = smoothed_points[indices[i][0]]

        results.append({
            'light_x': light[0],
            'light_y': light[1],
            'closest_x': closest_point[0],
            'closest_y': closest_point[1],
            'distance': distances[i][0]

        })
    
    return pd.DataFrame(results)

    
def find_closest_ann_balltree(smoothed_path, traffic_lights):

    from sklearn.neighbors import BallTree
    import pandas as pd

    smoothed_points = smoothed_path[['x_smooth_ma', 'y_smooth_ma']].values
   
    #Preprocessing: Build the ball tree from the smoothed path points
    path_tree = BallTree(smoothed_points)
    
    #Querying: Find the nearest neighbor for each traffic light
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
            'closest_y': closest_point[1],
            'distance': distances[i]
        })
    
    return pd.DataFrame(results)


