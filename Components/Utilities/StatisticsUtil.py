

def calculate_path_length(smoothed_vehicle_path_df, x_index = 'x_smooth_ma', y_index = 'y_smooth_ma'):
    
    import numpy as np
    
    points = smoothed_vehicle_path_df[[x_index, y_index]].values
    
    #difference between consecutive points (Delta X, Delta Y)
    deltas = points[1:] - points[:-1]
    
    #calculates sqrt(dx^2 + dy^2) for each segment.
    segment_lengths = np.linalg.norm(deltas, axis=1)
    
    total_length = np.sum(segment_lengths)
    
    return total_length


def calculate_avg_segment_distance(vehicle_path_df, x_index = "x_smooth_ma", y_index = 'y_smooth_ma'):
    
    
    total_length = calculate_path_length(vehicle_path_df, x_index , y_index)

    num_segments = len(vehicle_path_df)
    
    if num_segments == 0:
        return 0.0 # Handle case with 0 or 1 point
    
    average_distance = total_length / num_segments
    
    return average_distance


def search_performance_comparison_test(vehicle_path_df, traffic_lights_arr, subset_size_incrementor = 10):

    import time
    from Components.Utilities import NearestNeighborUtil


    full_path_size = len(vehicle_path_df)
    
    # --- Define the data subsets for the test ---
    # Create subsets of the data in steps of 10 points
    test_sizes = list(range(10, full_path_size + 1, subset_size_incrementor))
    if full_path_size % 10 != 0:
        test_sizes.append(full_path_size)
    test_sizes = sorted(list(set(test_sizes))) # Remove duplicates and sort
    
    print(test_sizes)
    brute_force_runtimes = []
    ann_kdtree_runtimes = []
    ann_balltree_runtimes = []
    
    # --- Run the performance test loop ---
    print("Running performance test...")
    for size in test_sizes:
        # Get the current data subset
        subset_df = vehicle_path_df.iloc[:size]
    
        #print(subset_df)

        # Measure Brute-Force runtime
        start_time = time.time()
        closest_points_results_brute_force_df = NearestNeighborUtil.run_brute_force_alg(subset_df, traffic_lights_arr)
        brute_force_runtimes.append(time.time() - start_time)

        # Measure ANN KDTree runtime
        start_time = time.time()
        closest_points_results_ann_kdtree_df = NearestNeighborUtil.run_ann_kdtree_alg(subset_df, traffic_lights_arr)
        ann_kdtree_runtimes.append(time.time() - start_time)

        # Measure ANN BallTree runtime
        start_time = time.time()
        closest_points_results_ann_balltree_df = NearestNeighborUtil.run_ann_balltree_alg(subset_df, traffic_lights_arr)
        ann_balltree_runtimes.append(time.time() - start_time)

        #print(f"Tested with {size} points.")
    
    return test_sizes, brute_force_runtimes, ann_kdtree_runtimes, ann_balltree_runtimes
