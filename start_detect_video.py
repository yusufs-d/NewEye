from services.DetectVideo import DetectVideo

user_input = input("Enter video path: ")


detect = DetectVideo(user_input,threshold=0.6)

try:
    detect.detect_with_distance()

except:
    print("Error! Please check your video path")