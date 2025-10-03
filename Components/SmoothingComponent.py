from Components.Utilities import DataPreparationUtil
from Components.Utilities import SmoothingUtil

def initiate_smoothing_analysis():

    import matplotlib.pyplot as plt

    cleaned_vehicle_path_df , traffic_lights_array = DataPreparationUtil.load_and_clean()

    smoothed_ma_vehicle_path_df = SmoothingUtil.smooth_path_moving_avg(cleaned_vehicle_path_df, 10)

    smoothed_spline_x, smoothed_spline_y = SmoothingUtil.smooth_path_interpolation_on_moving_avg(
        smoothed_ma_vehicle_path_df)

    
    # --- Plotting the original and smoothed paths ---
    plt.figure(figsize=(10, 8))
    plt.gca().set_aspect('equal', adjustable='box') 

    # Plot the original path
    plt.plot(cleaned_vehicle_path_df['x_coords'], 
             cleaned_vehicle_path_df['y_coords'],
            'o-', label='Original Path', color='gray', alpha=0.5)

    # Plot the moving average smoothed path
    plt.plot(smoothed_ma_vehicle_path_df['x_smooth_ma'], 
             smoothed_ma_vehicle_path_df['y_smooth_ma'], 
             's--', label='Moving Average', color='purple')

    # Plot the spline interpolation smoothed path
    plt.plot(smoothed_spline_x, smoothed_spline_y,
            '-', label='Spline Interpolation (on MA)', color='green')

    # Add labels and legend
    plt.title('Vehicle Path Smoothing (MA-based Spline)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Outputs/smoothed_path_ma_spline.png')
    plt.show()

    return


