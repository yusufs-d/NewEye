# NewEye
Atilim University 2023-2024 Spring Senior Project Group-10
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
4. Activate the virtual environment
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
python3 detect.py --modeldir "sample_model"
```
6. You can start taking screenshot process from the video you provided. Change "your_video.mp4" part with the path of your video.
```
python3 detect_video.py --modeldir "sample_model" --video "your_video.mp4"
```
6. Now you can start distance determination process
```
python3 services/DefineDistance.py
```
   
