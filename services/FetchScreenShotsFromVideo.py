import os
import time
import cv2

class VideoPlayer:
    def __init__(self,videoPath : str, resolution : tuple, targetDirectory: str, frequencySecond: int = 10):
        self.videoPath = videoPath
        self.resolution = resolution
        self.targetDirectory = targetDirectory
        self.videoName = self.videoPath.split(".")[0]
    
    def play(self):
        try:
            
            video = cv2.VideoCapture(self.videoPath)

            width = self.resolution[0]
            height = self.resolution[1]

            os.makedirs(self.targetDirectory,exist_ok=True)

            counter = 0

            cv2.namedWindow(self.videoName, cv2.WINDOW_NORMAL)
            video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            print("Selected video is playing...")
            start_time = time.time()
            while True:
                ret, frame = video.read()

                if not ret:
                    break

                if frame is None:
                    break
                
                current_time = time.time() - start_time

                if (current_time >= 10):
                    counter += 1
                    file = os.path.join(self.targetDirectory,f"{self.videoName}_screenshot_{counter}.png")
                    cv2.imwrite(file,frame)
                    start_time = time.time()

                cv2.imshow(self.videoName, frame)
                

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video.release()
            cv2.destroyAllWindows()
        except:
            print("Selected video couldn't be played due to an error! Please check your video path.")

