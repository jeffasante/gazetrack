# GazeTrack

A real-time eye tracking system for gaze estimation using computer vision. Built with OpenCV and supports both Haar cascade and dlib-based face detection.

## Features

- Real-time face and eye detection
- Pupil tracking with temporal smoothing
- Gaze estimation with calibration support
- Interactive calibration interface
- Validation system for accuracy measurement
- Multiple detection backends (Haar cascades, dlib)


## Installation

1. Clone the repository:
```bash
git clone https://github.com/jeffasante/gazetrack
cd gazetrack
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up model files:
```bash
python setup_models.py
```
   Or manually download and place in `models/` directory (see Model Files section below)

4. Install the package:
```bash
pip install -e .
```

## Quick Start

```python
from system import GazePySystem

# Initialize system
system = GazePySystem(screen_width=1920, screen_height=1080)

# Run interactive demo
system.run_interactive()
```

## Usage

### Interactive Mode

Run the interactive demo with real-time visualization:

```bash
python system.py
```

### Controls

- `d` - Toggle debug visualizations
- `c` - Start calibration process
- `v` - Start validation (requires calibration)
- `l` - Load saved calibration model
- `s` - Save current calibration model
- `q` or `ESC` - Quit application

### Calibration Process

1. Press `c` to start calibration
2. Look at each calibration point as it appears
3. System collects samples automatically
4. Model trains after all points are collected
5. Press `s` to save the calibration for future use

### Basic API Usage

```python
from system import GazePySystem

# Initialize
system = GazePySystem()
system.cam.start()

# Process single frame
frame = system.cam.get_frame()
result, data = system.process_frame(frame)

# Get gaze coordinates if available
if 'gaze_coords' in data:
    gaze_x, gaze_y = data['gaze_coords']  # Normalized coordinates (-1 to 1)
    screen_x, screen_y = system.gaze_estimator.gaze_to_screen_coordinates(gaze_x, gaze_y)

# Cleanup
system.cam.release()
```

## Technical Details

### Detection Backends

**Haar Cascades** (default):
- Fast performance
- Lower accuracy
- Works without additional models

**Dlib** (optional):
- Higher accuracy
- Requires shape_predictor_68_face_landmarks.dat
- More robust facial landmark detection

### Mathematical Foundations

#### 1. Pupil Detection Algorithm

The pupil detection uses morphological image processing:

```
1. Convert eye region to grayscale: I_gray = cvtColor(I_rgb, GRAY)
2. Gaussian blur: I_blur = GaussianBlur(I_gray, σ=7)
3. Dynamic thresholding: T = min(I_blur) + 25
4. Binary thresholding: I_binary = threshold(I_blur, T)
5. Morphological operations:
   - Opening: I_open = opening(I_binary, kernel_ellipse_5x5)
   - Closing: I_clean = closing(I_open, kernel_ellipse_5x5)
6. Contour analysis with circularity filter:
   circularity = 4π × area / perimeter²
   valid_pupil = circularity > 0.6
```

#### 2. Gaze Feature Extraction

For each eye, we extract normalized features:

**Horizontal Ratio:**
```
h_ratio = 1.0 - (pupil_x - inner_corner_x) / (outer_corner_x - inner_corner_x)
```

**Vertical Ratio:**
```
v_ratio = (pupil_y - upper_lid_y) / (lower_lid_y - upper_lid_y)
```

**Normalized Displacement:**
```
eye_center_x = (inner_corner_x + outer_corner_x) / 2
eye_center_y = (upper_lid_y + lower_lid_y) / 2

norm_disp_x = -1 × (pupil_x - eye_center_x) / (eye_width / 2)
norm_disp_y = (pupil_y - eye_center_y) / (eye_height / 2)
```

**Pupil Radius:**
```
pupil_radius = √(pupil_area / π)
```

#### 3. Temporal Smoothing

**Exponential Moving Average:**
```
smoothed_value(t) = α × current_value(t) + (1-α) × smoothed_value(t-1)
where α = 0.4 (smoothing factor)
```

**Weighted Historical Average (Gaze):**
```
weights = linspace(0.1, 1.0, history_length)
normalized_weights = weights / sum(weights)

gaze_x_smooth = Σ(w_i × gaze_x_i) for i in history
gaze_y_smooth = Σ(w_i × gaze_y_i) for i in history
```

#### 4. Coordinate Transformations

**Eye Region to Absolute Coordinates:**
```
absolute_x = eye_bbox_x + relative_x
absolute_y = eye_bbox_y + relative_y
```

**Gaze to Screen Coordinates:**
```
normalized_x = (gaze_x + 1) / 2  # Convert from [-1,1] to [0,1]
normalized_y = (gaze_y + 1) / 2

screen_x = normalized_x × screen_width
screen_y = normalized_y × screen_height
```

#### 5. Calibration Mathematics

**Feature Vector Construction:**
```
X = [left_h_ratio, left_v_ratio, left_norm_disp_x, left_norm_disp_y, left_pupil_radius,
     right_h_ratio, right_v_ratio, right_norm_disp_x, right_norm_disp_y, right_pupil_radius]
```

**Polynomial Feature Expansion:**
```
Φ(X) = [1, x₁, x₂, ..., xₙ, x₁², x₁x₂, ..., xₙ²]  (degree=2)
```

**Ridge Regression Model:**
```
minimize: ||ΦW - Y||² + α||W||²
where:
- Φ is the polynomial feature matrix
- W is the weight matrix
- Y is the target screen coordinates
- α = 0.5 (regularization parameter)
```

**Gaze Estimation:**
```
gaze_point = Φ(X_new) × W_trained
```

#### 6. Geometric Gaze Estimation (Fallback)

When calibration is unavailable:

```
avg_horizontal = (left_h_ratio + right_h_ratio) / 2
avg_vertical = (left_v_ratio + right_v_ratio) / 2

gaze_x = (avg_horizontal - 0.5) × 3.0
gaze_y = (avg_vertical - 0.5) × 3.0

gaze_x_clipped = clip(gaze_x, -1, 1)
gaze_y_clipped = clip(gaze_y, -1, 1)
```

#### 7. Validation Metrics

**Euclidean Distance Error:**
```
error_pixels = √[(target_x - estimated_x)² + (target_y - estimated_y)²]
```

**Accuracy Statistics:**
```
mean_error = (1/n) × Σ(error_i) for i=1 to n
std_deviation = √[(1/n) × Σ(error_i - mean_error)²]
max_error = max(error_i) for i=1 to n
```

### Gaze Estimation

The system uses a two-stage approach:

1. **Geometric Method**: Basic gaze estimation based on pupil position relative to eye landmarks
2. **Regression Model**: Machine learning model trained during calibration for improved accuracy

### Calibration

- 9-point calibration grid by default
- Collects multiple samples per point for robustness
- Uses polynomial features and Ridge regression
- Supports saving/loading trained models

## Performance

Typical performance on modern hardware:
- 15-30 FPS with Haar cascades
- 10-20 FPS with dlib detection
- Accuracy: 50-200 pixels after calibration (depends on setup)

## Model Files and References

### Required Models

1. **Haar Cascade Classifiers**
   - `haarcascade_eye.xml`
   - Source: OpenCV library (automatically installed)
   - Location: Usually in `cv2.data.haarcascades` directory
   - Reference: Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features.

2. **Dlib Face Landmark Predictor**
   - `shape_predictor_68_face_landmarks.dat`
   - Size: 99.7MB
   - Download sources:
     - Official: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
     - Alternative: https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
   - Reference: Kazemi, V., & Sullivan, J. (2014). One millisecond face alignment with an ensemble of regression trees.

### Model Setup Script


```python
import cv2
import os
import urllib.request
import bz2

def setup_models():
    os.makedirs('models', exist_ok=True)
    
    # Copy Haar cascades from OpenCV
    cv2_data = cv2.data.haarcascades
    face_cascade = os.path.join(cv2_data, 'haarcascade_frontalface_default.xml')
    eye_cascade = os.path.join(cv2_data, 'haarcascade_eye.xml')
    
    import shutil
    shutil.copy(face_cascade, 'models/')
    shutil.copy(eye_cascade, 'models/')
    
    # Download dlib model if not exists
    if not os.path.exists('models/shape_predictor_68_face_landmarks.dat'):
        print("Downloading dlib facial landmarks model...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, 'models/shape_predictor_68_face_landmarks.dat.bz2')
        
        with bz2.BZ2File('models/shape_predictor_68_face_landmarks.dat.bz2') as f:
            with open('models/shape_predictor_68_face_landmarks.dat', 'wb') as out:
                out.write(f.read())
        
        os.remove('models/shape_predictor_68_face_landmarks.dat.bz2')
        print("Model setup complete!")

if __name__ == "__main__":
    setup_models()
```

Run this to fetch models.
```python
python get_models.py
```


### Citations

```bibtex
@inproceedings{viola2001rapid,
  title={Rapid object detection using a boosted cascade of simple features},
  author={Viola, Paul and Jones, Michael},
  booktitle={Proceedings of the 2001 IEEE computer society conference on computer vision and pattern recognition},
  volume={1},
  pages={I--I},
  year={2001},
  organization={IEEE}
}

@inproceedings{kazemi2014one,
  title={One millisecond face alignment with an ensemble of regression trees},
  author={Kazemi, Vahid and Sullivan, Josephine},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4032--4039},
  year={2014}
}

@article{hansen2009eye,
  title={In the eye of the beholder: a survey of models for eyes and gaze},
  author={Hansen, Dan Witzner and Ji, Qiang},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={32},
  number={3},
  pages={478--500},
  year={2009},
  publisher={IEEE}
}

@inproceedings{wood2015rendering,
  title     = {Rendering of Eyes for Eye-Shape Registration and Gaze Estimation},
  author    = {Wood, Erroll and Baltrusaitis, Tadas and Zhang, Xucong and Sugano, Yusuke and Robinson, Peter and Bulling, Andreas},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {3756--3764},
  year      = {2015}
}

```

## Related Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Dlib Library](http://dlib.net/)
- [Eye Tracking Research Survey](https://www.sciencedirect.com/science/article/pii/S1389041718303048)
- [Gaze Estimation Papers](https://paperswithcode.com/task/gaze-estimation)

## License

Licensed under the Apache License, Version 2.0. See the LICENSE file for details.
