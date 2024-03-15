import sys
import time
import os
from detect_video import detect_and_take_screenshot
from services.DefineDistance import DefineDistance

video = input("Enter video path: ")
if os.path.exists(video):
    detect_and_take_screenshot(video)
else:
    print("Video path provided can not be found! Check your video path.")
    sys.exit(1)

print("Object detection and taking screenshots process ended!")
print("Defining distances process is starting...")
time.sleep(5)
try:
    define_distance_process = DefineDistance("ScreenShots")
    define_distance_process.start_process()
except:
    print("There is an error while starting 'Define Distance' process ")

