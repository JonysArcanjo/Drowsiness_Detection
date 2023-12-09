[![Medium](https://img.shields.io/badge/Medium-%23000000.svg?style=for-the-badge&logo=Medium&logoColor=white)](https://medium.com/@jonysarcanjo) [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&labelColor=blue)](https://www.linkedin.com/in/jonysarcanjo/) [![OpenCV](https://img.shields.io/badge/OpenCV-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/) [![Python](https://img.shields.io/badge/Python-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Computer Vision](https://img.shields.io/badge/Computer%20Vision-%23000000?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAWElEQVQ4T2NkIAoYf/58+Q8YRiQRMyAaMgPjEGMEAzMDI8RAbGgYGBgEGMEIiAyNAAYxAyMDI8RAbGgYJoNjYGBgYBBlZmZmoIJRYPj/8+fPf2HkYGBgAADJAxMFtDq5TAAAAABJRU5ErkJggg==&logoColor=white)](YOUR_LINK_HERE)



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)







# Drowsiness Detection 
[**Assista ao Vídeo**](https://www.canva.com/design/DAF2gSZamd8/m4PF97PTUsC4Fgj7-uvMAA/watch)



![CAPA Medium (2)](https://github.com/JonysArcanjo/Drowsiness_Detection/assets/48812740/793c9578-9d4d-47f7-b294-3952995da6e4)


This project uses Python computer vision to monitor a user's eyes and detect signs of drowsiness. This technology has the potential to prevent accidents related to drowsiness, such as those that occur while driving.


# Package Import

The necessary packages are imported at the beginning of the code. The main packages include:

- `dlib`: A library containing machine learning algorithms and image processing tools, including face detection.
- `cv2 (OpenCV)`: A computer vision library that contains functions for processing images and videos.
- `imutils`: A library that simplifies some basic image processing operations with OpenCV.
- `numpy`: A library used for scientific computing with Python.
- `playsound`: A library for playing sounds.
- `argparse`: A library for creating user-friendly command line interfaces.
- `scipy.spatial`: A module for calculating distances and performing operations in n-dimensional spaces.
- `matplotlib.pyplot`: Used for plotting data.

## Project Structure

The structure of the project is as follows:

```
.
├── Readme.md
├── alarm.wav
├── environment.yaml
└── main.py
└── shape_predictor_68_face_landmarks.dat
```

Description of the files:

* `alarm.wav`: Alarm sound that will be played when signs of fatigue are detected.
* `main.py`: Main Python script that contains the logic of the fatigue detector.
* `requirements.txt`: A file that lists the dependencies required to run the program.
* `shape_predictor_68_face_landmarks.dat`: Data file used by the dlib facial landmarks detector.


## Custom Functions

Two custom functions are defined at the beginning of the program. The `play_alarm(path)` function plays an alarm sound. The `calculate_ear(eye)` function computes the Eye Aspect Ratio (EAR). The EAR is a numeric value that indicates the degree of eye openness. When a person blinks, the EAR decreases.

## Command Line Arguments

The program accepts two command line arguments: "-a" to turn the sound alarm on or off, and "-w" to select the camera that will be used to capture the video.

## Initial Setup

The program defines some constants, such as the EAR threshold that indicates a blink (`EAR_THRESHOLD`) and the number of consecutive frames where the EAR must be below the threshold to trigger the alarm (`NUM_CONSEC_FRAMES`). It also initializes some variables and loads the dlib face detector and facial landmarks predictor.


## Video Processing

The program enters a continuous loop, where each iteration of the loop processes a single frame of the video. For each frame, the program detects present faces, identifies the facial features for the face region, extracts the coordinates of the left and right eyes, and calculates the EAR for both eyes.

The program draws the contours of the eyes on the original image using the `cv2.drawContours()` method. The EAR ratio is displayed on a real-time graph with each iteration.

## Drowsiness Detection

If the EAR is below the threshold for a set number of consecutive frames, the program considers the user to be becoming drowsy and triggers an alarm.

If the EAR returns to being above the threshold, the frame counter and the alarm are reset.

The code also displays the calculated EAR on the video frame to aid in debugging and adjusting the appropriate EAR thresholds.

## Termination

The video processing loop continues until the 'q' key is pressed. When this happens, the program ends and finalizes the video stream.

## Literary References

SOUKUPOVA, Tereza; CECH, Jan. [Real-time eye blink detection using facial landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf). In: 21st computer vision winter workshop, Rimske Toplice, Slovenia. 2016.

## License

This project is licensed under the MIT License
