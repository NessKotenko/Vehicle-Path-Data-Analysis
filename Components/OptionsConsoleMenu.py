from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem
import matplotlib.pyplot as plt
import sys

from Components import ExploratoryDataComponent
from Components import SmoothingComponent
from Components import NearestNeighborComponent
from Components import StatisticsComponent



def action_one():
    print("Executing Initial Exploratory Data Analysis...")
    ExploratoryDataComponent.initiate_data_analysis()

def action_two():
    print("Executing Smoothing Algorithms Analysis...")
    SmoothingComponent.initiate_smoothing_analysis()
    
def action_three():
    print("Executing Closest Points To Traffic Lights Search Analysis...")
    NearestNeighborComponent.init_nearest_neighbor_search_analysis()

def action_four():
    print("Executing Points Distances & Search Algorithms Performance Testing Statistics...")
    StatisticsComponent.initiate_path_and_search_perf_stats_analysis()



def console_menu():
    menu = ConsoleMenu("Path Processing Menu", "Select an option")
    
    menu.append_item(FunctionItem("Perform Initial Exploratory Data Analysis", action_one))
    menu.append_item(FunctionItem("Perform Smoothing Algorithms Analysis", action_two))
    menu.append_item(FunctionItem("Perform Closest Points To Traffic Lights Search", action_three))
    menu.append_item(FunctionItem("Perform Distance & Search Algorithms Performance Statistics", action_four))

    menu.show()
    
    print("Cleaning up Matplotlib figures...")
    plt.close('all')
    print("Exited successfully.")