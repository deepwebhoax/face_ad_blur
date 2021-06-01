# Advertising and face blurring in real time

This repository contains an app that detects and blurs advertising and faces on images taking input from your webcam. We are using MTCNN model to detect faces and YOLO model to detect advertising.

## How to use the app

Install the required packages:

    pip install -r requirements.txt
    

Run the app:

    python qtapp.py
 
## How to train YOLO

To train YOLOv5 you can use [this notebook](https://colab.research.google.com/drive/1JknNAFHaWk6yeKmMrW2fKmXd35u0FgVF?usp=sharing).

    
   

## References:

YOLOv5 to detect ads: https://github.com/ultralytics/yolov5

MTCNN to detect faces: https://github.com/timesler/facenet-pytorch

Advertising dataset: http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/Readme.pdf
