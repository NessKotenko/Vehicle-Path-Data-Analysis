from Components.Utilities import DataPreparationUtil


def initiate_data_analysis():

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Set print options to suppress scientific notation and set a high precision
    np.set_printoptions(suppress=True, precision=5)

    # Check vehicle path data:
    # Get the directory of the current script
    
    # Construct the full, absolute path to the data file
    vehicle_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'Data Files', 'Original', 'vehicle_path.npz')


    loaded_vehicle_path_data = np.load(vehicle_file_path)
    print(loaded_vehicle_path_data)

    loaded_vehicle_path_array = loaded_vehicle_path_data['path']
    print(loaded_vehicle_path_array)


    # Draw simple graph out of data:

    import numpy as np

    loaded_vehicle_path_data = np.load(vehicle_file_path)
    loaded_vehicle_path_array = loaded_vehicle_path_data['path']

    x_coords = loaded_vehicle_path_array[:, 0]
    y_coords = loaded_vehicle_path_array[:, 1]

    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')
    plt.title('Vehicle Path Raw')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('./Outputs/vehicle_path_raw_data.png')

    plt.show()



    #Clean Data:
    cleaned_vehicle_path_array = DataPreparationUtil.remove_nan_inf_rows(loaded_vehicle_path_array)

    print(cleaned_vehicle_path_array)


    # Draw simple graph out of clean data:
    
    x_cleaned_coords = cleaned_vehicle_path_array[:, 0]
    y_cleaned_coords = cleaned_vehicle_path_array[:, 1]


    plt.figure(figsize=(8, 6))
    plt.plot(x_cleaned_coords, y_cleaned_coords, marker='o', linestyle='-', color='b')
    plt.title('Vehicle Path Cleaned')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('./Outputs/vehicle_path_clean_data.png')
    plt.show()


    # Load traffic lights data
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full, absolute path to the data file
        
    traffic_lights_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'Data Files', 'Original', 'traffic_lights.npz')
    

    loaded_traffic_lights_positions_data = np.load(traffic_lights_file_path)
    loaded_traffic_lights_positions_array = loaded_traffic_lights_positions_data['positions']
    traffic_lights_x_coords = loaded_traffic_lights_positions_array[:, 0]
    traffic_lights_y_coords = loaded_traffic_lights_positions_array[:, 1]
    
    
    # scatter traffic lights with cleaned vehicle path plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_cleaned_coords, y_cleaned_coords, marker='o', linestyle='-', color='b')
    plt.scatter(traffic_lights_x_coords, traffic_lights_y_coords,
                label='Traffic Lights', color='red', marker='X', s=100)
    plt.title('Vehicle Path Cleaned & Traffic Lights')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('./Outputs/vehicle_path_clean_data.png')
    plt.show()
    

    return





