from services.DetectVideo import DetectVideo


detect = DetectVideo("/Users/yusufs/Desktop/Videos/ankara.mp4",threshold=0.6)

detect.detect_with_distance()