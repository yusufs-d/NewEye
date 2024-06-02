# Author: Evan Juras, Yusuf Salih Demir
# Date: 10/27/19, 04/29/24


import os
import threading
import cv2
from joblib import load
import numpy as np
import time
from threading import Thread
import importlib.util
import pandas as pd
from services.GetRegion import get_region, get_region_with_middle
from services.Speak import play_audio
from services.VideoStream import VideoStream


class StartProcess:
    def __init__(self):

        play_audio("opening.wav")
        play_audio("loading.mp3")
        play_audio("contact.mp3")

        self.MODEL_NAME = "sample_model"
        self.GRAPH_NAME = "detect.tflite"
        self.LABELMAP_NAME = "labelmap.txt"
        self.min_conf_threshold = float(0.5)
        self.resW, self.resH = [1280,720]
        self.imW, self.imH = int(self.resW), int(self.resH)
        self.use_TPU = False
        self.logRegModel,self.label_encoder = load("distance_model/distance_prediction_logReg-model.joblib")

        # Get path to current working directory
        self.CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH,self.MODEL_NAME,self.GRAPH_NAME)

        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH,self.MODEL_NAME,self.LABELMAP_NAME)

        # Initialize video stream
        self.videostream = VideoStream(resolution=(self.imW,self.imH),framerate=30).start()
        time.sleep(1)

    
    def start_detection_process(self):

        # Import TensorFlow libraries
        # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if self.use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (GRAPH_NAME == 'detect.tflite'):
                GRAPH_NAME = 'edgetpu.tflite'       



        # Load the label map
        with open(self.PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if self.use_TPU:
            interpreter = Interpreter(model_path=self.PATH_TO_CKPT,
                                    experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(self.PATH_TO_CKPT)
        else:
            interpreter = Interpreter(model_path=self.PATH_TO_CKPT)

        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        outname = output_details[0]['name']

        if ('StatefulPartitionedCall' in outname): # This is a TF2 model
            boxes_idx, classes_idx, scores_idx = 1, 3, 0
        else: # This is a TF1 model
            boxes_idx, classes_idx, scores_idx = 0, 1, 2


        # Initialize frame rate calculation
        frame_rate_calc = 1
        freq = cv2.getTickFrequency()


        play_audio("ready.mp3")

        # Create a named window
        cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
        # Resize the window
        cv2.resizeWindow('Object detector', 640, 480)
        
        #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        while True:

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            # Grab frame from video stream
            frame1 = self.videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * self.imH)))
                    xmin = int(max(1,(boxes[i][1] * self.imW)))
                    ymax = int(min(self.imH,(boxes[i][2] * self.imH)))
                    xmax = int(min(self.imW,(boxes[i][3] * self.imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    rect_width = abs(xmax - xmin)
                    rect_height = abs(ymax - ymin)
                    rect_area = round(rect_height * rect_width * 0.001,3)
                    region = get_region((xmin,ymin,xmax,ymax),self.imW,self.imH)
                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    new_data = pd.DataFrame({'ObjectName': [object_name],
                                                'Size': [rect_area],
                                                'Region': [region]})
                    object_name_to_sound_11 = object_name.strip() + "11" + ".mp3"
                    object_name_to_sound_12 = object_name.strip() + "12" + ".mp3"
                    object_name_to_sound_1 = object_name.strip() + "1" + ".mp3"
                    

                    try:
                        new_data['ObjectName'] = self.label_encoder.transform(new_data['ObjectName'])
                        predicted_result = self.logRegModel.predict(new_data)
                        predicted_result_text = ""
                        if predicted_result[0] == 1:
                            if not (object_name == "person" and rect_width > 600): 
                                print(f"{object_name} detected at region {region}. Size is : {rect_area}")
                                predicted_result_text = "Close"
                                region_with_middle = get_region_with_middle((xmin,ymin,xmax,ymax),self.imW,self.imH)

                                if region_with_middle == 1 or region_with_middle == 3:
                                    threading.Thread(target=play_audio,args=(object_name_to_sound_11,)).start()
                                elif region_with_middle == 2 or region_with_middle == 4:
                                    threading.Thread(target=play_audio,args=(object_name_to_sound_1,)).start()
                                elif region_with_middle == 'middle':
                                    threading.Thread(target=play_audio,args=(object_name_to_sound_12,)).start()
                    except:
                        new_data = pd.DataFrame({'ObjectName': ["person"],
                                'Size': [rect_area],
                                'Region': [region]})
                        new_data['ObjectName'] = self.label_encoder.transform(new_data['ObjectName'])
                        predicted_result = self.logRegModel.predict(new_data)
                        predicted_result_text = ""

                        if predicted_result[0] == 1:
                            
                            if not (object_name == "person" and rect_width > 600):

                                print(f"{object_name} detected at region {region}. Size is : {rect_area}")

                                predicted_result_text = "Close"
                                region_with_middle = get_region_with_middle((xmin,ymin,xmax,ymax),self.imW,self.imH)

                                if region_with_middle == 1 or region_with_middle == 3:
                                    threading.Thread(target=play_audio,args=(object_name_to_sound_11,)).start()
                                elif region_with_middle == 2 or region_with_middle == 4:
                                    threading.Thread(target=play_audio,args=(object_name_to_sound_1,)).start()
                                elif region_with_middle == 'middle':
                                    threading.Thread(target=play_audio,args=(object_name_to_sound_12,)).start()        
                    cv2.putText(frame, f"{object_name} {predicted_result_text}", (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text


            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)



            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        # Clean up
        cv2.destroyAllWindows()
        self.videostream.stop()
