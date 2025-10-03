import sys
import subprocess
import importlib.util



def check_and_install_packages(requirements_file="requirements.txt"):
    
    # Check if pip is installed
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: pip is not installed. Please install pip to proceed.")
        sys.exit(1)

    with open(requirements_file, 'r') as f:
        required_packages = [line.strip() for line in f if line.strip()]

    print("Checking for required packages...")
    packages_to_install = []
    for package in required_packages:
        if importlib.util.find_spec(package.split('==')[0].split('>')[0].split('<')[0]) is None:
            packages_to_install.append(package)

    if packages_to_install:
        print("The following packages are missing and will be installed:")
        for pkg in packages_to_install:
            print(f"- {pkg}")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *packages_to_install])
            print("All missing packages installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during package installation: {e}")
            sys.exit(1)
    else:
        print("All required packages are already installed.")


def main():
    
    try:
        import consolemenu
        from Components import OptionsConsoleMenu
        OptionsConsoleMenu.console_menu()


    except ModuleNotFoundError as e:
        print("Import Returned Error:", e)
        run_alternative()
    
    except Exception as e:
        print("An unexpected error occurred:", e)
        sys.exit(1)



def run_alternative():
    
    from Components import ExploratoryDataComponent
    from Components import SmoothingComponent
    from Components import NearestNeighborComponent
    from Components import StatisticsComponent

    ExploratoryDataComponent.initiate_data_analysis()
    SmoothingComponent.initiate_smoothing_analysis()
    NearestNeighborComponent.init_nearest_neighbor_search_analysis()
    StatisticsComponent.initiate_path_and_search_perf_stats_analysis()

        
if __name__ == "__main__":
    check_and_install_packages()
    main()