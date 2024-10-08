# NewEye: Object Detection and Voice Feedback for Visually Impaired Individuals

*Developed by **Yusuf Salih Demir***

NewEye is an object detection and voice feedback system developed to help visually impaired individuals better perceive the objects around them. Using TensorFlow Lite, it detects nearby objects and audibly announces the object's name and direction.

<img width="1270" alt="Screenshot 2024-06-06 at 18 42 08" src="https://github.com/yusufs-d/NewEye/assets/74401288/ce8fbbe8-72a9-4207-a2f8-bdfb6fa8bf70">
<img width="1270" alt="Screenshot 2024-06-06 at 18 49 11" src="https://github.com/yusufs-d/NewEye/assets/74401288/2bb6681d-467e-4deb-90e9-0777c09c40e0">
<img width="1270" alt="Screenshot 2024-06-06 at 18 35 50" src="https://github.com/yusufs-d/NewEye/assets/74401288/27dc4f51-3489-447f-9fef-4327273d142a">



## Features

- **Real-Time Object Detection:** Quickly and efficiently detects objects using TensorFlow Lite.
- **Proximity Detection:** Uses a custom machine learning algorithm to determine the proximity of objects.
- **Region Detection:** Divides the screen into four main regions to identify the object's location.
- **Voice Feedback:** Audibly announces the detected object's name and direction.
- **User-Friendly:** Simple and intuitive voice interface for easy use. It can be used with any camera compatible with Pc.

## Technical Details.

- **OpenCV-Python and TensorFlow Lite:** Utilized for object detection processes.
- **Proximity Algorithm:** Determines object proximity based on parameters such as pixel size on the screen, the region of the screen, and the object's name.
- **Voice Feedback:** Provides audible information about the detected object using Python's voice feedback libraries.

## Contribute

 I am continuously developing this project and welcome contributions from the community. If you would like to contribute, please send a pull request or report an issue.



I developed this project to enhance the quality of life for visually impaired individuals, and I greatly value your feedback. Please help me by contributing or providing feedback!

# Usage
## Follow these steps for initial usage
1. Clone this repo to your computer
```
git clone https://github.com/yusufs-d/NewEye.git   
```
2. Set your current directory as the repo you have cloned
```
cd NewEye
```
3. Create a virtual environment
```
pip install virtualenv
```
```
python -m venv newEyeEnv
```
4. Activate the virtual environment (Do this step every restart your terminal/cmd)
- **For Windows**
  ```
  newEyeEnv\Scripts\activate
  ```
- **For Mac and Linux**
  ```
  source newEyeEnv/bin/activate
  ```
5. Install the required libs for the project
```
pip install -r requirements.txt
```
6. You are all set! Now you can run the project
```
python3 main.py
```
6. You can also start defining video detection process if you want to detect objects from a video
```
python3 start_detect_video.py
```
6. Just enter the path of your video 
```
Enter video path:
```
   
