# Emojify - Quick Reference Guide

## Key Features Summary

### 1. **Real-Time Emotion Recognition**
- Live webcam feed processing
- 11 emotion classes (0-10)
- Automatic emoji overlay on detected faces

### 2. **Dataset Creation**
- **Webcam**: Capture images in real-time (press 'c' to capture)
- **From Files**: Process existing face images
- Automatic face detection, alignment, and preprocessing

### 3. **Deep Learning Pipeline**
- CNN model training and retraining
- Model evaluation and accuracy metrics
- TensorBoard visualization

### 4. **Face Processing**
- 68-point facial landmark detection
- Face alignment for consistency
- Feature masking (eyes, eyebrows, nose, mouth)

### 5. **Transparent Emoji Blending**
- Alpha channel support
- Natural overlay on faces

---

## Data Storage Quick Reference

| Data Type | Format | Location | Description |
|-----------|--------|----------|-------------|
| **Raw Images** | JPEG | dataset/{0-10}/ | Original training images |
| **Processed Images** | JPEG | new_dataset/{0-10}/ | Aligned & preprocessed faces |
| **Training Data** | Pickle | train_images, train_labels | 83.3% of dataset |
| **Test Data** | Pickle | test_images, test_labels | 8.3% of dataset |
| **Validation Data** | Pickle | val_images, val_labels | 8.3% of dataset |
| **CNN Model** | HDF5 (.h5) | cnn_model_keras.h5 | Trained neural network |
| **Emojis** | PNG (RGBA) | emojis/{0-10}.png | Overlay images |
| **Training Logs** | TensorBoard | logs/timestamp/ | Training metrics |
| **Face Landmarks** | Binary | shape_predictor_68_face_landmarks.dat | dlib model |

---

## Workflow Steps

### Step 1: Create Dataset
```
create_dataset_webcam.py  OR  create_dataset_from_faces.py
    ↓
Saves to: new_dataset/{emotion_class}/
```

### Step 2: Prepare Data
```
load_images.py (enter dataset folder: new_dataset)
    ↓
Creates: train_images, train_labels, test_images, test_labels, val_images, val_labels
```

### Step 3: Train Model
```
train_cnn_keras.py
    ↓
Saves: cnn_model_keras.h5
Logs: logs/timestamp/
```

### Step 4: Run Application
```
emojify.py
    ↓
Loads: cnn_model_keras.h5, emojis/*.png
Real-time: Webcam → Face detection → Emotion prediction → Emoji overlay
```

---

## Data Flow (Simplified)

**Collection** → **Preprocessing** → **Training** → **Inference**

1. **Collection**: Webcam/Images → Face detection → Alignment → Save to folders
2. **Preprocessing**: Load folders → Shuffle → Split (83/8/8) → Pickle files
3. **Training**: Load pickle → Build CNN → Train → Save .h5 model
4. **Inference**: Webcam → Face detection → CNN prediction → Emoji blend → Display

---

## File Purposes

| File | Purpose |
|------|---------|
| **emojify.py** | Main application - real-time emotion recognition with emoji overlay |
| **recognize.py** | Face recognition without emoji (shows emotion class number) |
| **create_dataset_webcam.py** | Capture training images from webcam |
| **create_dataset_from_faces.py** | Process existing face images for training |
| **load_images.py** | Convert dataset to train/test/val pickle files |
| **train_cnn_keras.py** | Train CNN from scratch |
| **retrain_cnn_keras.py** | Retrain existing model with new data |
| **compute_accuracy.py** | Evaluate model accuracy on training data |
| **preprocess_img.py** | Face masking and landmark utilities |
| **blend.py** | Transparent emoji overlay on faces |
| **display_all_faces.py** | Visualize dataset samples |

---

## Data Split Ratios

- **Training**: 83.3% (5/6 of total data)
- **Test**: 8.3% (half of remaining 1/6)
- **Validation**: 8.3% (half of remaining 1/6)

---

## Model Input/Output

**Input**: 100×100 grayscale face image (aligned, masked)
**Output**: 11 probabilities (one per emotion class)
**Prediction**: Class with highest probability (0-10)

---

## Dependencies

- OpenCV (cv2)
- dlib
- TensorFlow/Keras
- NumPy
- imutils
- scikit-learn
- pickle (built-in)

---

## Key Parameters

- **Image size**: 100×100 pixels
- **Face alignment**: 250×250 (then resize to 100×100)
- **Emotion classes**: 11 (0-10)
- **CNN**: Conv2D(32) → MaxPool → Dense(1024) → Dropout(0.6) → Dense(11)
- **Training**: 15 epochs, batch size 100, SGD optimizer
