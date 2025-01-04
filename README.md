# Car Crash Detection Using Dashcam Footage  

## Overview  
This project focuses on car crash detection using dashcam footage. It explores and compares traditional AI methods and deep learning approaches to enhance road safety by enabling faster emergency alerts and providing reliable accident evidence.

## Dataset  
The **Car Crash Dataset (CCD)** from Kaggle was used, comprising real traffic accident video frames captured by dashcams.  

### Key Details:  
- **Size**: 1500 videos, 50 frames each, totaling 75,000 frames.  
- **Labels**:  
  - `0`: No car crash.  
  - `1`: Car crash.  
- **Attributes**:  
  - Timing: Day or night.  
  - Weather: Normal, snowy, or rainy.  
  - Ego-involvement: Whether the ego-vehicle was involved in the accident.  

Each video frame is accompanied by a CSV annotation file containing labels and environmental attributes.

## Preprocessing  
To prepare the data for analysis, the following steps were applied:  
1. **Resizing**: Adjusting frames to a uniform size.  
2. **Grayscale Conversion**: Simplifying images to single-channel intensity.  
3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhancing image contrast.  
4. **CSV Refinement**: Dropping unnecessary columns for cleaner annotations.

## Methods  
Two types of methods were evaluated:  

### Traditional Methods  
- **SIFT + Random Forest**  
- **SIFT + Logistic Regression**

### Deep Learning Models  
- **VGG-19**  
- **MobileNetV3**

## Results  
The models were evaluated for accuracy, precision, recall, and their ability to handle diverse environmental conditions. Results demonstrated the effectiveness of deep learning models in capturing complex patterns compared to traditional methods.

<img width="916" alt="image" src="https://github.com/user-attachments/assets/b5e46234-661a-46df-8d48-47a9f6c67e8a" />
