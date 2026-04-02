# Emojify Project - Complete Documentation

## Project Overview
**Emojify** is a real-time facial expression recognition system that detects human emotions and overlays corresponding emojis on faces using computer vision and deep learning.

---

## Key Features

### 1. **Real-Time Facial Expression Recognition**
   - Detects faces in live webcam feed
   - Recognizes 11 different emotional expressions (0-10)
   - Overlays matching emoji on detected faces in real-time

### 2. **Dataset Creation Tools**
   - **Webcam Dataset Creator**: Capture training images directly from webcam
   - **Face Dataset Creator**: Process existing face images for training
   - Automatic face alignment and preprocessing

### 3. **Deep Learning CNN Model**
   - Custom Convolutional Neural Network architecture
   - Training and retraining capabilities
   - Model evaluation and accuracy computation

### 4. **Advanced Face Processing**
   - 68-point facial landmark detection using dlib
   - Face alignment for consistent feature extraction
   - Facial feature masking (eyes, eyebrows, nose, mouth)

### 5. **Transparent Emoji Blending**
   - Alpha channel support for smooth emoji overlay
   - Natural blending with background

---

## System Architecture

### Components:
1. **Data Collection Module** (`create_dataset_webcam.py`, `create_dataset_from_faces.py`)
2. **Data Preprocessing Module** (`load_images.py`, `preprocess_img.py`)
3. **Model Training Module** (`train_cnn_keras.py`, `retrain_cnn_keras.py`)
4. **Model Evaluation Module** (`compute_accuracy.py`)
5. **Real-Time Recognition Module** (`emojify.py`, `recognize.py`)
6. **Utility Modules** (`blend.py`, `display_all_faces.py`)

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION PHASE                         │
└─────────────────────────────────────────────────────────────────────┘

    Webcam Input                    Existing Face Images
         │                                   │
         ├──────────────┬────────────────────┤
         │              │                    │
         ▼              ▼                    ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│create_dataset│  │create_dataset│  │   Manual     │
│  _webcam.py  │  │_from_faces.py│  │  Collection  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┴──────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Face Detection      │
              │ (dlib detector)     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ 68-Point Landmark   │
              │ Detection           │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Face Alignment      │
              │ (FaceAligner)       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Feature Masking     │
              │ (eyes, nose, mouth) │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Resize to 100x100   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   dataset/ or       │
              │   new_dataset/      │
              │   ├── 0/ (emotion)  │
              │   ├── 1/            │
              │   ├── ...           │
              │   └── 10/           │
              └─────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING PHASE                          │
└─────────────────────────────────────────────────────────────────────┘

              ┌─────────────────────┐
              │   load_images.py    │
              │  (Data Loader)      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Load all images     │
              │ from dataset/       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Shuffle dataset     │
              │ (3 times)           │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Split Dataset:      │
              │ • Train: 83.3%      │
              │ • Test: 8.3%        │
              │ • Validation: 8.3%  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Pickle Files:       │
              │ • train_images      │
              │ • train_labels      │
              │ • test_images       │
              │ • test_labels       │
              │ • val_images        │
              │ • val_labels        │
              └─────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING PHASE                            │
└─────────────────────────────────────────────────────────────────────┘

              ┌─────────────────────┐
              │ train_cnn_keras.py  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Load Pickle Files   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ CNN Architecture:   │
              │ • Conv2D (32)       │
              │ • BatchNorm         │
              │ • MaxPooling2D      │
              │ • Flatten           │
              │ • Dense (1024)      │
              │ • Dropout (0.6)     │
              │ • Dense (11 classes)│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Training Process    │
              │ • Optimizer: SGD    │
              │ • Loss: Categorical │
              │   Cross-Entropy     │
              │ • Epochs: 15        │
              │ • Batch Size: 100   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Save Best Model     │
              │ cnn_model_keras.h5  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ TensorBoard Logs    │
              │ logs/timestamp/     │
              └─────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME INFERENCE PHASE                         │
└─────────────────────────────────────────────────────────────────────┘

    Webcam Feed
         │
         ▼
┌──────────────────┐
│  emojify.py      │
│  (Main App)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Load CNN Model   │
│ Load Emojis      │
│ (0.png - 10.png) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Capture Frame    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Face Detection   │
│ (dlib)           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 68-Point         │
│ Landmarks        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Create Mask      │
│ (preprocess_img) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Face Alignment   │
│ & Masking        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Resize 100x100   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ CNN Prediction   │
│ (11 classes)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Get Predicted    │
│ Emotion Class    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Select Emoji     │
│ (emojis/N.png)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Blend Emoji      │
│ (blend.py)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Display Result   │
└──────────────────┘
```

---

## Data Storage Structure

### Directory Structure:
```
EvilPort2-emojify-9370401/
│
├── dataset/                    # Original training dataset
│   ├── 0/                      # Emotion class 0
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── 1/                      # Emotion class 1
│   ├── 2/
│   └── ... (up to 10)
│
├── new_dataset/                # Processed/aligned dataset
│   ├── 0/
│   ├── 1/
│   └── ... (up to 10)
│
├── emojis/                     # Emoji overlay images
│   ├── 0.png                   # Emoji for emotion 0
│   ├── 1.png
│   └── ... (up to 10.png)
│
├── logs/                       # TensorBoard training logs
│   └── timestamp/
│
├── Pickle Files (Root):        # Preprocessed data
│   ├── train_images            # Training images (binary)
│   ├── train_labels            # Training labels (binary)
│   ├── test_images             # Test images (binary)
│   ├── test_labels             # Test labels (binary)
│   ├── val_images              # Validation images (binary)
│   └── val_labels              # Validation labels (binary)
│
├── Models:
│   └── cnn_model_keras.h5      # Trained CNN model (HDF5)
│
└── Assets:
    └── shape_predictor_68_face_landmarks.dat  # dlib model
```

---

## Data Storage Details

### 1. **Image Dataset Storage**
- **Format**: JPEG (.jpg)
- **Size**: 100x100 pixels (grayscale)
- **Organization**: Folder-based classification (folder name = emotion label)
- **Location**: `dataset/` and `new_dataset/`

### 2. **Preprocessed Data Storage**
- **Format**: Python Pickle (binary serialization)
- **Data Type**: NumPy arrays (float16 for images, uint8 for labels)
- **Files**:
  - `train_images`, `train_labels` (83.3% of data)
  - `test_images`, `test_labels` (8.3% of data)
  - `val_images`, `val_labels` (8.3% of data)

### 3. **Model Storage**
- **Format**: HDF5 (.h5)
- **File**: `cnn_model_keras.h5`
- **Contains**: 
  - Model architecture
  - Trained weights
  - Optimizer state
  - Training configuration

### 4. **Emoji Assets**
- **Format**: PNG with alpha channel (RGBA)
- **Location**: `emojis/`
- **Naming**: `{emotion_class}.png` (0.png to 10.png)

### 5. **Training Logs**
- **Format**: TensorBoard event files
- **Location**: `logs/timestamp/`
- **Contains**: Loss curves, accuracy metrics, model graphs

---

## Emotion Classes

The system recognizes **11 emotion classes** (0-10):
- Each class represents a different facial expression
- Examples might include: Happy, Sad, Angry, Surprised, Neutral, etc.
- Emojis are mapped 1:1 with emotion classes

---

## Technical Specifications

### Image Processing:
- **Input Size**: Variable (from webcam or files)
- **Processing Size**: 250x250 (during alignment)
- **Model Input Size**: 100x100x1 (grayscale)
- **Face Detection**: dlib HOG-based frontal face detector
- **Landmark Detection**: 68-point facial landmarks

### CNN Architecture:
```
Input (100x100x1)
    ↓
Conv2D (32 filters, 5x5)
    ↓
BatchNormalization
    ↓
MaxPooling2D (10x10, stride=10)
    ↓
Flatten
    ↓
Dense (1024 units, ReLU)
    ↓
BatchNormalization
    ↓
Dropout (0.6)
    ↓
Dense (11 units, Softmax)
    ↓
Output (11 emotion probabilities)
```

### Training Parameters:
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Learning Rate**: 0.01 (default)
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 100
- **Epochs**: 15 (default)
- **Data Split**: 83.3% train, 8.3% test, 8.3% validation

---

## Workflow Summary

### Phase 1: Data Collection
1. Capture images using webcam or process existing images
2. Detect faces and extract 68 facial landmarks
3. Align faces to standard position
4. Create mask highlighting facial features
5. Save preprocessed images in labeled folders

### Phase 2: Data Preparation
1. Load all images from dataset
2. Shuffle data randomly (3x for good mixing)
3. Split into train/test/validation sets
4. Serialize to pickle files for fast loading

### Phase 3: Model Training
1. Load preprocessed data from pickle files
2. Build CNN architecture
3. Train model with callbacks (checkpointing, logging)
4. Save best model based on validation accuracy
5. Evaluate on validation set

### Phase 4: Real-Time Inference
1. Capture webcam frame
2. Detect faces in frame
3. Extract and align face
4. Preprocess (mask + resize)
5. Predict emotion using CNN
6. Load corresponding emoji
7. Blend emoji onto face with transparency
8. Display result in real-time

---

## Workflow Summary

### Phase 1: Data Collection
1. Capture images using webcam or process existing images
2. Detect faces and extract 68 facial landmarks
3. Align faces to standard position
4. Create mask highlighting facial features
5. Save preprocessed images in labeled folders

### Phase 2: Data Preparation
1. Load all images from dataset
2. Shuffle data randomly (3x for good mixing)
3. Split into train/test/validation sets
4. Serialize to pickle files for fast loading

### Phase 3: Model Training
1. Load preprocessed data from pickle files
2. Build CNN architecture
3. Train model with callbacks (checkpointing, logging)
4. Save best model based on validation accuracy
5. Evaluate on validation set

### Phase 4: Real-Time Inference
1. Capture webcam frame
2. Detect faces in frame
3. Extract and align face
4. Preprocess (mask + resize)
5. Predict emotion using CNN
6. Load corresponding emoji
7. Blend emoji onto face with transparency
8. Display result in real-time

---

## Performance Considerations

1. **Real-time Processing**: Optimized for 30+ FPS
2. **Memory Management**: Pickle files deleted after loading to save RAM
3. **Model Efficiency**: Lightweight CNN for fast inference
4. **Face Alignment**: Ensures consistent feature extraction
5. **Batch Processing**: Efficient training with batch size 100

---

## Future Enhancement Possibilities

1. Support for multiple faces simultaneously
2. Additional emotion classes
3. Mobile deployment
4. Real-time emotion tracking/analytics
5. Custom emoji upload feature
6. Video file processing
7. Emotion intensity detection

---

## Python 3.12 Compatibility

All code has been updated to work with Python 3.12:
- TensorFlow 2.x Keras integration
- Modern OpenCV API compatibility
- Updated optimizer parameters
- Proper exception handling
