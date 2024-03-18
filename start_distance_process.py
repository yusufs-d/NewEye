import sys
import time
import os
from detect_video import detect_and_take_screenshot
from services.DefineDistance import DefineDistance
from services.JsonControl import JsonControl
from services.ResetToDefault import reset

jsonObj = JsonControl()
if jsonObj.check_control() == 0:
    video = input("Enter video path: ")
    if os.path.exists(video):
        if os.path.exists("ScreenShots"):
            print("ScreenShots folder already exists. Please remove it before the precess")
            sys.exit(1)
        if os.path.exists("info.txt"):
            print("info.txt file already exists. Please remove it before the process")
            sys.exit(1)
    
        detect_and_take_screenshot(video)
        jsonObj.increase_control()
    

    else:
        print("Video path provided can not be found! Check your video path.")
        sys.exit(1)

    print("Object detection and taking screenshots process ended!")
    print("Defining distances process is starting...")
    time.sleep(5)

    
    define_distance_process = DefineDistance(os.path.join("services","ScreenShots"))
    define_distance_process.start_process()


elif jsonObj.check_control() == 1:
    print("Defining distances process is starting...")
    time.sleep(5)

    define_distance_process = DefineDistance(os.path.join("services","ScreenShots"))
    define_distance_process.start_process()

else:
    decision = input("Looks like 'define distance process' has completed. Do you want to start the whole process again?\nBE CAREFUL: Your ScreenShots folder and Excel File will be DELETED! (yes/no): ")
    if decision == "yes":

        reset()

        print("You will start define distance process again. Please run this program again with 'python3 start_distance_process.py'")
    
    elif decision == "no":
        print("Action canceled by the user. Exiting...")
        sys.exit(1)
    else:
        print("Invalid input. Exiting...")
        sys.exit(1)

