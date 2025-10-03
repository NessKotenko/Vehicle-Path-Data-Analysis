

def smooth_path_moving_avg(vehicle_path_df, window_size = 5):
    
    vehicle_path_df['x_smooth_ma'] = vehicle_path_df['x_coords'].rolling(window=window_size).mean()
    vehicle_path_df['y_smooth_ma'] = vehicle_path_df['y_coords'].rolling(window=window_size).mean()
    smoothed_vehicle_path_df = vehicle_path_df.dropna()

    return smoothed_vehicle_path_df


def smooth_path_interpolation_on_moving_avg(df_ma_cleaned, lines = 3000):
    
    import numpy as np
    from scipy.interpolate import interp1d

    # Use the cleaned moving average points for interpolation
    f_x = interp1d(df_ma_cleaned['x_smooth_ma'], df_ma_cleaned['y_smooth_ma'], kind='cubic')
    x_new = np.linspace(df_ma_cleaned['x_smooth_ma'].min(), df_ma_cleaned['x_smooth_ma'].max(), lines)
    y_smooth_spline = f_x(x_new)

    return x_new, y_smooth_spline
