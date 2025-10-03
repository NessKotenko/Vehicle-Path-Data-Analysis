from Components.Utilities import DataPreparationUtil
from Components.Utilities import NearestNeighborUtil
from Components.Utilities import SmoothingUtil


def init_nearest_neighbor_search_analysis():


    import matplotlib.pyplot as plt
    
    # --- Load, Clean & Create a new smoothed path---
    
    vehicle_path_df , traffic_lights_array = DataPreparationUtil.load_and_clean()

    smoothed_vehicle_path_df = SmoothingUtil.smooth_path_moving_avg(vehicle_path_df, window_size = 10)

    #Find closest points with brute force algorithm
    closest_points_results_brute_force_df = NearestNeighborUtil.run_brute_force_alg(smoothed_vehicle_path_df, traffic_lights_array)



    # Print results for bruteforce:
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
    plt.title('Closest Points on Smoothed Path (Brute-Force Search Algorithm)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Outputs/closest_points_brute_force.png')
    plt.show()






    #Find closest points with kdtree algorithm
    closest_points_results_ann_kdtree_df = NearestNeighborUtil.run_ann_kdtree_alg(smoothed_vehicle_path_df, traffic_lights_array)

    # Print results for KDTree:
    print("Closest points found using ANN KDTree Search Algorithm:")
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
    plt.savefig('./Outputs/closest_points_ann_kdtree.png')
    plt.show()






    #Find closest points with balltree algorithm
    closest_points_results_ann_balltree_df = NearestNeighborUtil.run_ann_balltree_alg(smoothed_vehicle_path_df, traffic_lights_array)

    # Print results for BallTree:
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
    plt.title('Closest Points on Smoothed Path (ANN BallTree Search Algorithm)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Outputs/closest_points_ann_balltree.png')
    plt.show()





    return



