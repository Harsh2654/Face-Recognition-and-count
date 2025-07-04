# Face-Recognition-and-count
# Face Recognition and Authorization System

## Overview
This project is a Face Recognition and Authorization System that detects faces using a webcam, counts the total number of faces detected, and identifies authorized faces based on a given dataset of images stored in the `dataset` folder. If a detected face does not match any of the authorized faces, it is marked as "Unknown."

## Features
- **Real-time Face Detection**: Uses a webcam to detect faces .
- **Face Counting**: Counts the total number of detected faces.
- **Face Recognition**: Matches detected faces with an authorized dataset.
- **Unauthorized Face Detection**: Labels faces not found in the dataset as "Unknown."
- **Dataset-based Authorization**: Compares detected faces with the images in the `dataset` folder.

## Technologies Used
- **Python**
- **OpenCV**
- **YOLOv5** (for object detection)
- **Face Recognition Library**
- **NumPy**
- dlib 

## Installation
1. Clone this repository:
   bash
   git clone https://github.com/yourusername/face-recognition-project.git
   cd face-recognition-project

2. Install the required dependencies:
   bash
   pip install opencv-python numpy face-recognition torch torchvision pandas
  
3. Place authorized face images in the `dataset` folder.

## Usage
1. Run the face recognition script:
   bash
   python face_recognition.py
   
2. The system will:
   - Detect faces from the webcam.
   - Compare them with the dataset.
   - Display recognized faces and mark unknown faces.
   - Show the total count of faces detected.
  
   - Log entry in Excel

![image](https://github.com/user-attachments/assets/eb46c146-ea7f-4fe7-a347-69c7c138d1c9)

