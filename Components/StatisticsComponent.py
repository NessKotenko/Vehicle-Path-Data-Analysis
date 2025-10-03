
def initiate_path_and_search_perf_stats_analysis():

    import matplotlib.pyplot as plt
    import os
    from Components.Utilities import DataPreparationUtil
    from Components.Utilities import SmoothingUtil
    from Components.Utilities import StatisticsUtil
    
    # --- Load, Clean & Create a new smoothed path ---   
    vehicle_path_df , traffic_lights_array = DataPreparationUtil.load_and_clean()
    smoothed_vehicle_path_df = SmoothingUtil.smooth_path_moving_avg(vehicle_path_df, window_size = 10)

    # Calculate the path length
    path_length = StatisticsUtil.calculate_path_length(smoothed_vehicle_path_df)

    print(f"The total length of the smoothed vehicle path is: {path_length:.4f} units")
    


    # --- Load, Clean & Smooth path ---   
    vehicle_path_df , traffic_lights_array = DataPreparationUtil.load_and_clean()
    smoothed_vehicle_path_df = SmoothingUtil.smooth_path_moving_avg(vehicle_path_df, window_size = 10)


    # Calculate the average distance between consecutive points on original data points
    avg_segment_distance = StatisticsUtil.calculate_avg_segment_distance(vehicle_path_df, 'x_coords', 'y_coords')

    print(f"The average distance between consecutive points of the original path data is: {avg_segment_distance:.6f} units")


    # Calculate the average distance between consecutive points on smooth path

    avg_smooth_segment_distance = StatisticsUtil.calculate_avg_segment_distance(smoothed_vehicle_path_df)
    
    print(f"The average distance between consecutive points of the smoothed path is: {avg_smooth_segment_distance:.6f} units")






    #--- Run Performance Comparison Test ---
    test_sizes, brute_force_runtimes, ann_kdtree_runtimes, ann_balltree_runtimes = StatisticsUtil.search_performance_comparison_test(smoothed_vehicle_path_df, traffic_lights_array)

    # --- Plot the performance results comparison ---
    plt.figure(figsize=(10, 6))

    plt.plot(test_sizes, brute_force_runtimes, 'o-', label='Brute-Force', color='red')
    plt.plot(test_sizes, ann_kdtree_runtimes, 's-', label='ANN (KDTree)', color='blue')
    plt.plot(test_sizes, ann_balltree_runtimes, '^-', label='ANN (BallTree)', color='green')

    plt.title('Algorithm Runtime Performance Comparison On Real Data')
    plt.xlabel('Number of Path Points')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.tight_layout()

    plt.grid(True)
    plt.savefig('./Outputs/orig_data_search_perf_comparison.png')

    plt.show()






    ##CREATE SYNTH DATA for stress-testing performance 
    response = DataPreparationUtil.create_synthetic_path_and_traffic_lights(path_points_count=30000, traffic_lights_count=500)
    print(response)

    #LOAD SYNTH DATA
    synth_vehicle_path_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'Data Files', 'Synthetic', 'vehicle_path_synthetic.npz')
    synth_traffic_lights_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'Data Files', 'Synthetic', 'traffic_lights_synthetic.npz')


    synth_vehicle_path_df , synth_traffic_lights_array = DataPreparationUtil.load_and_clean(synth_vehicle_path_filename, synth_traffic_lights_filename)

    synth_smoothed_vehicle_path_df = SmoothingUtil.smooth_path_moving_avg(synth_vehicle_path_df, window_size = 20)
    

    #VIEW SYNTH DATA

    # --- Plotting the original and smoothed paths ---
    plt.figure(figsize=(200, 20))
    plt.gca().set_aspect('equal', adjustable='box') 

    # Plot the original path directly from the DataFrame
    plt.plot(synth_vehicle_path_df['x_coords'], synth_vehicle_path_df['y_coords'], 'o-', label='Original Path', color='gray', alpha=0.5)

    # Plot the moving average smoothed path from the DataFrame
    plt.plot(synth_smoothed_vehicle_path_df['x_smooth_ma'], synth_smoothed_vehicle_path_df['y_smooth_ma'], 's--', label='Moving Average', color='purple')

    plt.scatter(synth_traffic_lights_array[:, 0], synth_traffic_lights_array[:, 1], label='Traffic Lights', color='red', marker='X', s=100)


    # Add labels and legend
    plt.title('Synthetic Vehicle Path Smoothed (from DataFrame)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    #plt.tight_layout()
    plt.savefig('./Outputs/synthetic_path_data_smoothed.png')
    plt.show()





    #PERFORMANCE TEST SYNTH DATA

    #--- Run Performance Comparison Test on synthetic data---
    synth_test_sizes, synth_brute_force_runtimes, synth_ann_kdtree_runtimes,synth_ann_balltree_runtimes = StatisticsUtil.search_performance_comparison_test(synth_smoothed_vehicle_path_df, synth_traffic_lights_array, 100)
    
    # --- Plot synthetic performance results ---
    plt.figure(figsize=(10, 6))

    plt.plot(synth_test_sizes, synth_brute_force_runtimes, 'o-', label='Brute-Force', color='red')
    plt.plot(synth_test_sizes, synth_ann_kdtree_runtimes, 's-', label='ANN (KDTree)', color='blue')
    plt.plot(synth_test_sizes, synth_ann_balltree_runtimes, '^-', label='ANN (BallTree)', color='green')

    plt.title('Algorithm Runtime Performance Comparison On Synthetic Data')
    plt.xlabel('Number of Path Points')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.tight_layout()

    plt.grid(True)
    plt.savefig('./Outputs/synth_data_search_perf_comparison.png')
    plt.show()





    return




