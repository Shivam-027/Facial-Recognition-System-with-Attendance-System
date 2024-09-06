# Facial Recognition System

## Overview

This project implements a facial recognition system using Histogram of Oriented Gradients (HOG) features and a Support Vector Classifier (SVC). The system can recognize faces in real-time via a webcam and mark attendance based on detected faces. It leverages OpenCV for image processing and face detection, `skimage` for HOG feature extraction, and `scikit-learn` for classification.

## Features

- Real-time face detection using OpenCV.
- HOG feature extraction for facial recognition.
- SVC classification to identify known individuals.
- Automatic attendance marking with timestamp.

## Usage

1. **Prepare your dataset:**

    Place your images in a directory. The image filenames should be in the format `name.jpg` where `name` is the label of the person. For example:
    ```
    dataset/
    ├── MS_Dhoni.jpg
    └── so on...
    ```

2. **Update file paths:**

    Edit the `image_path` and `attendance_file` variables in `testing.py` to point to your dataset directory and desired attendance file location.

3. **Run the script:**

    Execute the script to start the facial recognition and attendance marking process:

    The script will open a webcam feed, detect faces, recognize them, and display the name of the recognized person on the screen. It will also mark the attendance in the specified CSV file.

4. **Exit the script:**

    Press `q` in the webcam window to stop the script.

## Code Explanation

- **`load_images_and_labels(path)`**: Loads images from the specified directory, resizes them, converts them to grayscale, and encodes labels.
- **`extract_features(images)`**: Extracts HOG features from the images.
- **`train_classifier(features, labels)`**: Trains an SVC classifier using the extracted features and labels.
- **`mark_attendance(name)`**: Marks attendance by appending the name, time, and date to the specified CSV file.
- **Main loop**: Captures video from the webcam, detects faces, extracts features, predicts the label using the trained classifier, and marks attendance if recognized.
