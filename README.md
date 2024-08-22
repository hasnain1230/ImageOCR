# ImageOCR Project

## Overview
This project was developed as part of the Imaging and Multimedia course at Rutgers University. The goal was to create a simple Optical Character Recognition (OCR) system using traditional image processing and computer vision techniques.

## What it does
The ImageOCR system can recognize a set of predefined characters (a, d, m, n, o, p, q, r, u, w) from input images. It processes the images, extracts features, and classifies characters based on these features.

## How it works
1. **Image Processing**: The system starts by reading character images and applying various preprocessing steps, including binarization and component labeling.

2. **Feature Extraction**: For each character, we extract moment-based features, including Hu moments, which describe the shape characteristics of the character.

3. **Training**: The extracted features for each known character are stored in a database. These features are normalized to ensure consistent comparison.

4. **Classification**: When given a test image, the system extracts features using the same process and compares them to the stored database. It classifies the character based on the nearest neighbor in the feature space.

5. **Evaluation**: The project includes methods to evaluate the system's performance, including confusion matrix calculation.

## Technologies Used
- Python
- NumPy and SciPy for numerical operations
- scikit-image for image processing tasks
- Matplotlib for visualization

## Reflections
This project was a great opportunity to apply the concepts learned in the Imaging and Multimedia course. It provided hands-on experience with image processing techniques and feature extraction methods. While more advanced OCR systems exist, building this from scratch helped deepen my understanding of the fundamental principles behind character recognition.

## Future Improvements
Given more time, it would be interesting to expand the character set, implement more advanced feature extraction techniques, or explore machine learning approaches for classification.

